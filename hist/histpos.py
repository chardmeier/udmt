from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule, RMSProp, Adam, Momentum, AdaGrad)
from blocks.bricks import application, Initializable, Tanh
from blocks.bricks.recurrent import GatedRecurrent, LSTM, SimpleRecurrent
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.serialization import load_parameters
from dependency import conll_trees
from fuel.transformers import Batch, FilterSources, Mapping, Padding
from fuel.datasets import IndexableDataset
from fuel.schemes import ConstantScheme, ShuffledScheme
from hist.tagger import TagPredictor, WordEmbedding
from theano import tensor

import argparse
import collections
import fuel
import itertools
import logging
import math
import numpy
import pprint
import re
import sys
import theano


logger = logging.getLogger(__name__)


class HistPOSTagger(Initializable):
    def __init__(self, alphabet_size, seq_dimensions, pos_dimension, transition_type, **kwargs):
        super().__init__(**kwargs)

        embedder = WordEmbedding(alphabet_size, seq_dimensions, transition_type)
        predictor = TagPredictor(seq_dimensions[-1], pos_dimension)

        self.embedder = embedder
        self.predictor = predictor
        self.children = [embedder, predictor]

    @application
    def cost(self,
             pos_chars, pos_chars_mask, pos_word_mask, pos_targets,
             norm_chars, norm_chars_mask, norm_word_mask,
             hist_chars, hist_chars_mask, hist_word_mask):
        pos_encoded, pos_collected_mask = self.embedder.apply(pos_chars, pos_chars_mask, pos_word_mask)
        pos_cost = self.predictor.cost(pos_encoded, pos_collected_mask, pos_targets)

        norm_encoded, norm_collected_mask = self.embedder.apply(norm_chars, norm_chars_mask, norm_word_mask)
        hist_encoded, hist_collected_mask = self.embedder.apply(hist_chars, hist_chars_mask, hist_word_mask)

        # Input char sequences can have different length, but the number of words should always be the same.
        # Trim the matrices to the same size.
        min_length = tensor.minimum(norm_encoded.shape[0], hist_encoded.shape[0])
        collected_mask = hist_collected_mask[0:min_length, :]
        norm_enc_trunc = norm_encoded[0:min_length, :]
        hist_enc_trunc = hist_encoded[0:min_length, :]

        diff_cost = (collected_mask * tensor.sqr(norm_enc_trunc - hist_enc_trunc).sum(axis=2)).sum(axis=0).mean()

        return pos_cost, diff_cost

    @application
    def apply(self, chars, chars_mask, word_mask):
        encoded, collected_mask = self.embedder.apply(chars, chars_mask, word_mask)
        return self.predictor.apply(encoded)


def _transpose(data):
    return tuple(array.T if array.ndim == 2 else array.transpose(1, 0, 2) for array in data)


def _is_nan(log):
    return math.isnan(log.current_row['total_gradient_norm'])


def load_conll(infile, chars_voc=None, pos_voc=None):
    seqs = []
    for tr in conll_trees(infile):
        seqs.append([(n.token, n.pos) for n in tr[1:]])

    if chars_voc is None:
        chars = set(itertools.chain(*(t[0] for t in itertools.chain(*seqs))))
        chars_voc = {c: i + 4 for i, c in enumerate(sorted(chars))}
        chars_voc['<UNK>'] = 0
        chars_voc['<S>'] = 1
        chars_voc['</S>'] = 2
        chars_voc[' '] = 3

    if pos_voc is None:
        tags = set(t[1] for t in itertools.chain(*seqs))
        pos_voc = {c: i + 3 for i, c in enumerate(sorted(tags))}
        pos_voc['<UNK>'] = 0
        pos_voc['<S>'] = 1
        pos_voc['</S>'] = 2

    floatx_vals = numpy.arange(2, dtype=fuel.config.floatX)
    zero = floatx_vals[0]
    one = floatx_vals[1]

    words = []
    chars = []
    pos = []
    word_mask = []
    for snt in seqs:
        snt_words = [word_tag[0] for word_tag in snt]
        snt_chars = [chars_voc['<S>']]
        snt_pos = [pos_voc['<S>']]
        snt_word_mask = [one]
        for word, postag in snt:
            snt_chars.extend(chars_voc.get(c, chars_voc['<UNK>']) for c in word)
            snt_chars.append(chars_voc[' '])
            snt_word_mask.extend([zero] * len(word) + [one])
            snt_pos.append(pos_voc.get(postag, pos_voc['<UNK>']))
        snt_chars.append(chars_voc['</S>'])
        snt_pos.append(pos_voc['</S>'])
        snt_word_mask.append(one)

        words.append(snt_words)
        chars.append(snt_chars)
        pos.append(snt_pos)
        word_mask.append(snt_word_mask)

    data = collections.OrderedDict()
    data['pos_words'] = words
    data['pos_chars'] = chars
    data['pos_word_mask'] = word_mask
    data['pos_targets'] = pos

    return data, chars_voc, pos_voc


def load_historical(infile, chars_voc=None):
    floatx_vals = numpy.arange(2, dtype=fuel.config.floatX)
    zero = floatx_vals[0]
    one = floatx_vals[1]

    empty_line = re.compile(r'\s*$')
    seqs = []
    for key, group in itertools.groupby(infile, lambda l: empty_line.match(l)):
        if not key:
            seqs.append([l.rstrip('\n').split('\t') for l in group])

    if chars_voc is None:
        chars = set(itertools.chain(*(t[0] + t[1] for t in itertools.chain(*seqs))))
        chars_voc = {c: i + 4 for i, c in enumerate(sorted(chars))}
        chars_voc['<UNK>'] = 0
        chars_voc['<S>'] = 1
        chars_voc['</S>'] = 2
        chars_voc[' '] = 3

    all_hist = []
    all_norm = []
    all_hist_word_mask = []
    all_norm_word_mask = []
    for snt_pairs in seqs:
        hist_chars = [chars_voc['<S>']]
        norm_chars = [chars_voc['<S>']]
        hist_word_mask = [one]
        norm_word_mask = [one]
        for i, (hist, norm) in enumerate(snt_pairs):
            hist_word = list(chars_voc.get(c, chars_voc['<UNK>']) for c in hist)
            hist_chars.extend(hist_word)
            hist_chars.append(chars_voc[' '])
            hist_word_mask.extend([zero] * len(hist) + [one])
            norm_word = list(chars_voc.get(c, chars_voc['<UNK>']) for c in norm)
            norm_chars.extend(norm_word)
            norm_chars.append(chars_voc[' '])
            norm_word_mask.extend([zero] * len(norm) + [one])
        hist_chars[-1] = chars_voc['</S>']
        norm_chars[-1] = chars_voc['</S>']

        all_hist.append(hist_chars)
        all_norm.append(norm_chars)
        all_hist_word_mask.append(hist_word_mask)
        all_norm_word_mask.append(norm_word_mask)

    data = collections.OrderedDict()
    data['hist_chars'] = all_hist
    data['hist_word_mask'] = all_hist_word_mask
    data['norm_chars'] = all_norm
    data['norm_word_mask'] = all_norm_word_mask

    return data, chars_voc


def join_training_data(pos_data, hist_data):
    pos_size = len(list(pos_data.values())[0])
    hist_size = len(list(hist_data.values())[0])
    max_size = max(pos_size, hist_size)

    pos_sample = numpy.random.choice(pos_size, size=max_size, replace=True)
    hist_sample = numpy.random.choice(hist_size, size=max_size, replace=True)

    all_data = collections.OrderedDict()
    for key, val in pos_data.items():
        all_data[key] = [val[i] for i in numpy.nditer(pos_sample)]
    for key, val in hist_data.items():
        all_data[key] = [val[i] for i in numpy.nditer(hist_sample)]

    return IndexableDataset(all_data)


def train(pos_weight, postagger, dataset, num_batches, save_path, step_rule='original'):
    # Data processing pipeline

    # dataset.example_iteration_scheme = ShuffledScheme(dataset.num_examples, 10)
    data_stream = dataset.get_example_stream()
    data_stream = Batch(data_stream, iteration_scheme=ConstantScheme(10))
    data_stream = FilterSources(data_stream,
                                sources=('pos_chars', 'pos_word_mask', 'pos_targets',
                                         'norm_chars', 'norm_word_mask',
                                         'hist_chars', 'hist_word_mask'))
    data_stream = Padding(data_stream)
    data_stream = FilterSources(data_stream,
                                sources=('pos_chars', 'pos_chars_mask', 'pos_word_mask', 'pos_targets',
                                         'norm_chars', 'norm_chars_mask', 'norm_word_mask',
                                         'hist_chars', 'hist_chars_mask', 'hist_word_mask'))
    data_stream = Mapping(data_stream, _transpose)

    # Initialization settings
    postagger.weights_init = IsotropicGaussian(0.1)
    postagger.biases_init = Constant(0.0)
    postagger.push_initialization_config()

    # Build the cost computation graph
    pos_chars = tensor.lmatrix('pos_chars')
    pos_chars_mask = tensor.matrix('pos_chars_mask')
    pos_word_mask = tensor.matrix('pos_word_mask')
    pos_targets = tensor.lmatrix('pos_targets')
    norm_chars = tensor.lmatrix('norm_chars')
    norm_chars_mask = tensor.matrix('norm_chars_mask')
    norm_word_mask = tensor.matrix('norm_word_mask')
    hist_chars = tensor.lmatrix('hist_chars')
    hist_chars_mask = tensor.matrix('hist_chars_mask')
    hist_word_mask = tensor.matrix('hist_word_mask')

    pos_cost, diff_cost = postagger.cost(pos_chars, pos_chars_mask, pos_word_mask, pos_targets,
                                         norm_chars, norm_chars_mask, norm_word_mask,
                                         hist_chars, hist_chars_mask, hist_word_mask)
    batch_cost = pos_weight * pos_cost + (1.0 - pos_weight) * diff_cost
    batch_size = pos_chars.shape[1].copy(name="batch_size")
    cost = aggregation.mean(batch_cost, batch_size)
    cost.name = "cost"
    logger.info("Cost graph is built")

    # Give an idea of what's going on
    model = Model(cost)
    parameters = model.get_parameter_dict()
    logger.info("Parameters:\n" +
                pprint.pformat(
                    [(key, value.get_value().shape) for key, value
                     in parameters.items()],
                    width=120))

    # Initialize parameters
    for brick in model.get_top_bricks():
        brick.initialize()

    if step_rule == 'original':
        step_rule_obj = CompositeRule([StepClipping(10.0), Scale(0.01)])
    elif step_rule == 'rmsprop':
        step_rule_obj = RMSProp(learning_rate=.01)
    elif step_rule == 'rms+mom':
        step_rule_obj = CompositeRule([RMSProp(learning_rate=0.01), Momentum(0.9)])
    elif step_rule == 'adam':
        step_rule_obj = Adam()
    elif step_rule == 'adagrad':
        step_rule_obj = AdaGrad()
    else:
        raise ValueError('Unknow step rule: ' + step_rule)

    # Define the training algorithm.
    cg = ComputationGraph(cost)
    algorithm = GradientDescent(cost=cost, parameters=cg.parameters, step_rule=step_rule_obj)

    # Fetch variables useful for debugging
    max_pos_length = pos_chars.shape[0].copy(name="max_pos_length")
    max_norm_length = pos_chars.shape[0].copy(name="max_norm_length")
    max_hist_length = pos_chars.shape[0].copy(name="max_hist_length")
    report_pos_cost = pos_cost.copy(name='pos_cost')
    report_diff_cost = diff_cost.copy(name='diff_cost')
    # cost_per_character = aggregation.mean(
    #     batch_cost, batch_size * max_length).copy(
    #     name="character_log_likelihood")
    observables = [
         cost, report_pos_cost, report_diff_cost,
         batch_size, max_pos_length, max_hist_length, max_norm_length, # cost_per_character,
         algorithm.total_step_norm, algorithm.total_gradient_norm]
    # for name, parameter in parameters.items():
    #     observables.append(parameter.norm(2).copy(name + "_norm"))
    #     observables.append(algorithm.gradients[parameter].norm(2).copy(
    #         name + "_grad_norm"))

    # Construct the main loop and start training!
    average_monitoring = TrainingDataMonitoring(
        observables, prefix="average", every_n_batches=25)

    main_loop = MainLoop(
        model=model,
        data_stream=data_stream,
        algorithm=algorithm,
        extensions=[
            Timing(),
            TrainingDataMonitoring(observables, after_batch=True),
            average_monitoring,
            FinishAfter(after_n_batches=num_batches)
                # This shows a way to handle NaN emerging during
                # training: simply finish it.
                .add_condition(["after_batch"], _is_nan),
            # Saving the model and the log separately is convenient,
            # because loading the whole pickle takes quite some time.
            Checkpoint(save_path, every_n_batches=500,
                       save_separately=["model", "log"]),
            Printing(every_n_batches=25)])
    main_loop.run()


def predict(postagger, dataset, save_path, pos_voc):
    data_stream = dataset.get_example_stream()
    data_stream = Batch(data_stream, iteration_scheme=ConstantScheme(50))
    data_stream = Padding(data_stream, mask_sources=('chars', 'word_mask'))
    data_stream = FilterSources(data_stream, sources=('words', 'chars', 'chars_mask', 'word_mask'))
    data_stream = Mapping(data_stream, _transpose)

    chars = tensor.lmatrix('chars')
    chars_mask = tensor.matrix('chars_mask')
    word_mask = tensor.matrix('word_mask')
    pos, out_mask = postagger.apply(chars, chars_mask, word_mask)

    model = Model(pos)
    with open(save_path, 'rb') as f:
        model.set_parameter_values(load_parameters(f))

    tag_fn = theano.function(inputs=[chars, chars_mask, word_mask], outputs=[pos, out_mask])

    reverse_pos = {idx: word for word, idx in pos_voc.items()}

    for i_words, i_chars, i_chars_mask, i_word_mask in data_stream.get_epoch_iterator():
        o_pos, o_mask = tag_fn(i_chars, i_chars_mask, i_word_mask)
        o_pos_idx = numpy.argmax(o_pos, axis=-1)
        for sntno in range(o_pos_idx.shape[1]):
            words = i_words[sntno]
            tags = [reverse_pos[o_pos_idx[i, sntno]] for i in range(o_pos_idx.shape[0]) if o_mask[i, sntno]]
            # tags has begin and end tokens, words doesn't.
            for word, tag in zip(words, tags[1:-1]):
                print('%s\t%s' % (word, tag))
            print()


def evaluate(postagger, dataset, save_path, pos_voc):
    data_stream = dataset.get_example_stream()
    data_stream = Batch(data_stream, iteration_scheme=ConstantScheme(50))
    data_stream = FilterSources(data_stream, sources=('chars', 'word_mask', 'pos'))
    data_stream = Padding(data_stream, mask_sources=('chars', 'word_mask', 'pos'))
    data_stream = FilterSources(data_stream, sources=('chars', 'chars_mask', 'word_mask', 'pos'))
    data_stream = Mapping(data_stream, _transpose)

    chars = tensor.lmatrix('chars')
    chars_mask = tensor.matrix('chars_mask')
    word_mask = tensor.matrix('word_mask')
    pos_targets = tensor.lmatrix('pos_targets')

    pos, out_mask = postagger.apply(chars, chars_mask, word_mask)

    model = Model(pos)
    with open(save_path, 'rb') as f:
        model.set_parameter_values(load_parameters(f))

    tag_fn = theano.function(inputs=[chars, chars_mask, word_mask], outputs=[pos, out_mask])

    npos = len(pos_voc)

    # total = 0
    # matches = 0

    gold_table = numpy.zeros((npos,))
    pred_table = numpy.zeros((npos,))
    hit_table = numpy.zeros((npos,))

    for i_chars, i_chars_mask, i_word_mask, i_pos_idx in data_stream.get_epoch_iterator():
        o_pos, o_mask = tag_fn(i_chars, i_chars_mask, i_word_mask)
        o_pos_idx = numpy.argmax(o_pos, axis=-1)

        # total += o_mask.sum()
        # matches += ((i_pos_idx == o_pos_idx) * o_mask).sum()

        fmask = numpy.flatnonzero(o_mask)
        fgold = i_pos_idx.flatten()[fmask]
        fpredict = o_pos_idx.flatten()[fmask]

        gold_pos, gold_counts = numpy.unique(fgold, return_counts=True)
        gold_table[gold_pos] += gold_counts

        pred_pos, pred_counts = numpy.unique(fpredict, return_counts=True)
        pred_table[pred_pos] += pred_counts

        hit_pos, hit_counts = numpy.unique(fgold[fgold == fpredict], return_counts=True)
        hit_table[hit_pos] += hit_counts

    reverse_pos = {idx: word for word, idx in pos_voc.items()}

    total = gold_table[3:].sum()
    matches = hit_table[3:].sum()

    precision = hit_table / pred_table
    recall = hit_table / gold_table
    fscore = 2.0 * precision * recall / (precision + recall)

    print('Accuracy: %5d/%5d = %6f\n' % (matches, total, matches / total))
    for i in range(npos):
        print('%10s : P = %4d/%4d = %6f      R = %4d/%4d = %6f      F = %6f' %
              (reverse_pos[i],
               hit_table[i], pred_table[i], precision[i],
               hit_table[i], gold_table[i], recall[i],
               fscore[i]))


def main():
    parser = argparse.ArgumentParser(
        "POS tagger",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "mode", choices=["train", "predict", "eval"],
        help="The mode to run. In the `train` mode a model is trained."
             " In the `predict` mode a trained model is "
             " to used to tag the input text.")
    parser.add_argument(
        "taggermodel",
        help="The path to save the training process if the mode"
             " is `train` OR path to an `.tar` files with learned"
             " parameters if the mode is `predict`.")
    parser.add_argument(
        "histtrain",
        help="the historical training corpus (required for both training and prediction)")
    parser.add_argument(
        "postrain",
        help="the POS training corpus (required for both training and prediction)")
    parser.add_argument(
        "--pos-weight", default=0.5, type=float,
        help="Weight of pos_cost relative to diff_cost")
    parser.add_argument(
        "--num-batches", default=10000, type=int,
        help="Train on this many batches.")
    parser.add_argument(
        "--recurrent-type", choices=["rnn", "gru", "lstm"], default="rnn",
        help="The type of recurrent unit to use")
    parser.add_argument(
        "--step-rule", choices=["original", "rmsprop", "rms+mom", "adam", "adagrad"], default="original",
        help="The step rule for the search algorithm")
    parser.add_argument(
        "--test-file", required=False,
        help="The file to test on in prediction mode")
    args = parser.parse_args()

    with open(args.histtrain, 'r') as f:
        hist_data, chars_voc = load_historical(f)

    with open(args.postrain, 'r') as f:
        pos_data, _, pos_voc = load_conll(f, chars_voc=chars_voc)

    train_ds = join_training_data(pos_data, hist_data)

    tagger = HistPOSTagger(len(chars_voc), [50, 150, 51], len(pos_voc), args.recurrent_type)

    if args.mode == "train":
        num_batches = args.num_batches
        train(args.pos_weight, tagger, train_ds, num_batches, args.taggermodel, step_rule=args.step_rule)
    # elif args.mode == "predict":
    #     with open(args.test_file, 'r') if args.test_file is not None else sys.stdin as f:
    #         text_ds = load_vertical(f, chars_voc)
    #     test_enc = embed(embedder, text_ds, args.embedmodel)
    #     test_data = collections.OrderedDict()
    #     test_data['embeddings'] = test_enc
    #     test_ds = IndexableDataset(test_data)
    #     predict(tagger, test_ds, args.taggermodel, pos_voc)
    # elif args.mode == "eval":
    #     with open(args.test_file, 'r') if args.test_file is not None else sys.stdin as f:
    #         text_ds, pos, _ = load_conll(f, chars_voc, pos_voc=pos_voc)
    #     test_enc = embed(embedder, text_ds, args.embedmodel)
    #     test_data = collections.OrderedDict()
    #     test_data['embeddings'] = test_enc
    #     test_data['pos'] = pos
    #     test_ds = IndexableDataset(test_data)
    #     evaluate(tagger, test_ds, args.taggermodel, pos_voc)


if __name__ == '__main__':
    main()