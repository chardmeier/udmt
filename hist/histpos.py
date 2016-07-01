from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule, RMSProp, Adam, Momentum, AdaGrad)
from blocks.bricks import application, Initializable
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, IsotropicGaussian
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.roles import INPUT, WEIGHT
from blocks.serialization import load_parameters
from dependency import conll_trees
from fuel.transformers import Batch, FilterSources, Mapping, Padding
from fuel.datasets import IndexableDataset
from fuel.schemes import ConstantScheme
from hist.tagger import load_vertical, TagPredictor, WordEmbedding
from theano import tensor

import argparse
import collections
import fuel
import itertools
import json
import logging
import math
import numpy
import pickle
import pprint
import re
import sys
import theano


logger = logging.getLogger(__name__)


class Configuration:
    def load_json(self, infile):
        self.__dict__.update(json.load(infile))

    def dump_json(self):
        return json.dumps(self.__dict__, indent=2)


class HistPOSTaggerConfiguration(Configuration):
    def __init__(self, alphabet_size=None, pos_dimension=None):
        self.alphabet_size = alphabet_size
        self.pos_dimension = pos_dimension
        self.share_embedders = True
        self.recurrent_type = 'rnn'
        self.sequence_dims = [50, 150, 51]
        self.hidden_dims = []
        self.diff_loss = 'squared_error'


class TrainingConfiguration(Configuration):
    def __init__(self):
        self.histtrain_file = None
        self.postrain_file = None
        self.histval_file = None
        self.posval_file = None
        self.pos_weight = 0.5
        self.num_batches = 100000
        self.step_rule = 'original'
        self.l2_penalty = 0.0


class HistPOSTagger(Initializable):
    def __init__(self, net_config, **kwargs):
        super().__init__(**kwargs)

        hist_embedder = WordEmbedding(net_config.alphabet_size, net_config.sequence_dims, net_config.recurrent_type)
        self.children.append(hist_embedder)

        if net_config.share_embedders:
            norm_embedder = hist_embedder
        else:
            norm_embedder = WordEmbedding(net_config.alphabet_size, net_config.sequence_dims, net_config.recurrent_type)
            self.children.append(norm_embedder)

        predictor = TagPredictor([net_config.sequence_dims[-1]] + net_config.hidden_dims + [net_config.pos_dimension])
        self.children.append(predictor)

        self.diff_loss = net_config.diff_loss

        self.norm_embedder = norm_embedder
        self.hist_embedder = hist_embedder
        self.predictor = predictor

    @application
    def pos_cost(self, pos_chars, pos_chars_mask, pos_word_mask, pos_targets):
        pos_encoded, pos_collected_mask = self.norm_embedder.apply(pos_chars, pos_chars_mask, pos_word_mask)
        pos_cost = self.predictor.cost(pos_encoded, pos_collected_mask, pos_targets)

        return pos_cost

    @application
    def diff_cost(self,
                  norm_chars, norm_chars_mask, norm_word_mask,
                  hist_chars, hist_chars_mask, hist_word_mask):
        norm_encoded, norm_collected_mask = self.norm_embedder.apply(norm_chars, norm_chars_mask, norm_word_mask)
        hist_encoded, hist_collected_mask = self.hist_embedder.apply(hist_chars, hist_chars_mask, hist_word_mask)

        # Input char sequences can have different length, but the number of words should always be the same.
        # Trim the matrices to the same size.
        min_length = tensor.minimum(norm_encoded.shape[0], hist_encoded.shape[0])
        collected_mask = hist_collected_mask[0:min_length, :]
        norm_enc_trunc = norm_encoded[0:min_length, :]
        hist_enc_trunc = hist_encoded[0:min_length, :]

        if self.diff_loss == 'squared_error':
            diff_cost = (collected_mask * tensor.sqr(norm_enc_trunc - hist_enc_trunc).sum(axis=2)).sum(axis=0).mean()
        elif self.diff_loss == 'crossentropy':
            diff_cost = (collected_mask *
                         (norm_enc_trunc * tensor.log(hist_enc_trunc) +
                          ((1.0 - norm_enc_trunc) * tensor.log(1.0 - hist_enc_trunc))).sum(axis=2)).sum(axis=0).mean()
        else:
            raise ValueError

        return diff_cost

    @application
    def apply(self, embedder, chars, chars_mask, word_mask):
        if embedder == 'norm':
            embedder_net = self.norm_embedder
        elif embedder == 'hist':
            embedder_net = self.hist_embedder
        else:
            raise ValueError

        encoded, collected_mask = embedder_net.apply(chars, chars_mask, word_mask)
        return self.predictor.apply(encoded), collected_mask


def _transpose(data):
    # Flip the first two dimensions regardless of total dimensionality
    maxdim = max(array.ndim for array in data)
    if maxdim == 1:
        return data
    perm = list(range(maxdim))
    perm[0:2] = [1, 0]
    return tuple(array if array.ndim == 1 else array.transpose(perm[0:array.ndim]) for array in data)


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


def train(postagger, train_config, dataset, save_path, pos_validation_set=None, hist_validation_set=None):
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

    pos_cost = postagger.pos_cost(pos_chars, pos_chars_mask, pos_word_mask, pos_targets)
    pos_cost.name = 'pos_cost'

    diff_cost = postagger.diff_cost(norm_chars, norm_chars_mask, norm_word_mask,
                                    hist_chars, hist_chars_mask, hist_word_mask)
    diff_cost.name = 'diff_cost'

    unregularised_cost = train_config.pos_weight * pos_cost + (1.0 - train_config.pos_weight) * diff_cost
    unregularised_cost.name = 'unregularised_cost'

    if train_config.l2_penalty:
        cg_unreg = ComputationGraph(unregularised_cost)
        weights = VariableFilter(roles=[WEIGHT])(cg_unreg.variables)
        regularisation_term = sum(tensor.sqr(w).sum() for w in weights)
    else:
        regularisation_term = 0.0

    cost = unregularised_cost + regularisation_term
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

    if train_config.step_rule == 'original':
        step_rule_obj = CompositeRule([StepClipping(10.0), Scale(0.01)])
    elif train_config.step_rule == 'rmsprop':
        step_rule_obj = RMSProp(learning_rate=.01)
    elif train_config.step_rule == 'rms+mom':
        step_rule_obj = CompositeRule([RMSProp(learning_rate=0.01), Momentum(0.9)])
    elif train_config.step_rule == 'adam':
        step_rule_obj = Adam()
    elif train_config.step_rule == 'adagrad':
        step_rule_obj = AdaGrad()
    else:
        raise ValueError('Unknow step rule: ' + train_config.step_rule)

    # Define the training algorithm.
    cg = ComputationGraph(cost)
    algorithm = GradientDescent(cost=cost, parameters=cg.parameters, step_rule=step_rule_obj)

    # Fetch variables useful for debugging
    batch_size = pos_chars.shape[1].copy(name="batch_size")
    max_pos_length = pos_chars.shape[0].copy(name="max_pos_length")
    max_norm_length = norm_chars.shape[0].copy(name="max_norm_length")
    max_hist_length = hist_chars.shape[0].copy(name="max_hist_length")
    report_pos_cost = pos_cost.copy(name='pos_cost')
    report_diff_cost = diff_cost.copy(name='diff_cost')
    # cost_per_character = aggregation.mean(
    #     batch_cost, batch_size * max_length).copy(
    #     name="character_log_likelihood")
    observables = [
         cost, report_pos_cost, report_diff_cost,
         batch_size, max_pos_length, max_hist_length, max_norm_length,  # cost_per_character,
         algorithm.total_step_norm, algorithm.total_gradient_norm]
    # for name, parameter in parameters.items():
    #     observables.append(parameter.norm(2).copy(name + "_norm"))
    #     observables.append(algorithm.gradients[parameter].norm(2).copy(
    #         name + "_grad_norm"))

    # Construct the main loop and start training!
    average_monitoring = TrainingDataMonitoring(
        observables, prefix="average", every_n_batches=25)

    validation_set_monitoring = []
    if pos_validation_set:
        pos_val_data_stream = dataset.get_example_stream()
        pos_val_data_stream = Batch(pos_val_data_stream, iteration_scheme=ConstantScheme(100))
        pos_val_data_stream = FilterSources(pos_val_data_stream,
                                            sources=('pos_chars', 'pos_word_mask', 'pos_targets'))
        pos_val_data_stream = Padding(pos_val_data_stream)
        pos_val_data_stream = FilterSources(pos_val_data_stream,
                                            sources=('pos_chars', 'pos_chars_mask', 'pos_word_mask', 'pos_targets'))
        pos_val_data_stream = Mapping(pos_val_data_stream, _transpose)

        val_pos_cost = pos_cost.copy(name='val_pos_cost')

        pos_monitoring = DataStreamMonitoring([val_pos_cost], pos_val_data_stream)
        validation_set_monitoring.append(pos_monitoring)

    if hist_validation_set:
        hist_val_data_stream = dataset.get_example_stream()
        hist_val_data_stream = Batch(hist_val_data_stream, iteration_scheme=ConstantScheme(100))
        hist_val_data_stream = FilterSources(hist_val_data_stream,
                                             sources=('norm_chars', 'norm_word_mask',
                                                      'hist_chars', 'hist_word_mask'))
        hist_val_data_stream = Padding(hist_val_data_stream)
        hist_val_data_stream = FilterSources(hist_val_data_stream,
                                             sources=('norm_chars', 'norm_chars_mask', 'norm_word_mask',
                                                      'hist_chars', 'hist_chars_mask', 'hist_word_mask'))
        hist_val_data_stream = Mapping(hist_val_data_stream, _transpose)

        val_diff_cost = diff_cost.copy(name='val_diff_cost')

        hist_monitoring = DataStreamMonitoring([val_diff_cost], hist_val_data_stream)
        validation_set_monitoring.append(hist_monitoring)

    main_loop = MainLoop(
        model=model,
        data_stream=data_stream,
        algorithm=algorithm,
        extensions=[
                       Timing(),
                       TrainingDataMonitoring(observables, after_batch=True),
                       average_monitoring
                   ] + validation_set_monitoring + [
                       FinishAfter(after_n_batches=train_config.num_batches)
                           # This shows a way to handle NaN emerging during
                           # training: simply finish it.
                           .add_condition(["after_batch"], _is_nan),
                       # Saving the model and the log separately is convenient,
                       # because loading the whole pickle takes quite some time.
                       Checkpoint(save_path, every_n_batches=500,
                                  save_separately=["model", "log"]),
                       Printing(every_n_batches=25)
                   ])
    main_loop.run()


def predict(postagger, dataset, save_path, pos_voc, embedder='norm'):
    data_stream = dataset.get_example_stream()
    data_stream = Batch(data_stream, iteration_scheme=ConstantScheme(50))
    data_stream = Padding(data_stream, mask_sources=('chars', 'word_mask'))
    data_stream = FilterSources(data_stream, sources=('words', 'chars', 'chars_mask', 'word_mask'))
    data_stream = Mapping(data_stream, _transpose)

    chars = tensor.lmatrix('chars')
    chars_mask = tensor.matrix('chars_mask')
    word_mask = tensor.matrix('word_mask')
    pos, out_mask = postagger.apply(embedder, chars, chars_mask, word_mask)

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


def evaluate(postagger, dataset, save_path, pos_voc, embedder='norm'):
    data_stream = dataset.get_example_stream()
    data_stream = Batch(data_stream, iteration_scheme=ConstantScheme(50))
    data_stream = FilterSources(data_stream, sources=('pos_chars', 'pos_word_mask', 'pos_targets'))
    data_stream = Padding(data_stream, mask_sources=('pos_chars', 'pos_word_mask', 'pos_targets'))
    data_stream = FilterSources(data_stream, sources=('pos_chars', 'pos_chars_mask', 'pos_word_mask', 'pos_targets'))
    data_stream = Mapping(data_stream, _transpose)

    chars = tensor.lmatrix('pos_chars')
    chars_mask = tensor.matrix('pos_chars_mask')
    word_mask = tensor.matrix('pos_word_mask')

    pos, out_mask = postagger.apply(embedder, chars, chars_mask, word_mask)

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


def save_metadata(outfile, net_config, chars_voc, pos_voc):
    with open(outfile + '.meta', 'wb') as f:
        pickle.dump(f, net_config)
        pickle.dump(f, chars_voc)
        pickle.dump(f, pos_voc)


def load_metadata(infile):
    with open(infile + '.meta', 'rb') as f:
        net_config = pickle.load(f)
        chars_voc = pickle.load(f)
        pos_voc = pickle.load(f)
    return net_config, chars_voc, pos_voc


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
        "--train-config", required=False,
        help="Training configuration JSON file.")
    parser.add_argument(
        "--net-config", required=False,
        help="Network configuration JSON file (for training mode only).")
    parser.add_argument(
        "--embedder", choices=["norm", "hist"], default="norm",
        help="Embedder to use in prediction and evaluation mode if not shared.")
    parser.add_argument(
        "--test-file", required=False,
        help="The file to test on in prediction and evaluation mode")
    args = parser.parse_args()

    if args.mode == "train":
        train_config = TrainingConfiguration()
        if args.train_config is not None:
            with open(args.train_config, 'r') as f:
                train_config.load_json(f)

        with open(train_config.histtrain_file, 'r') as f:
            hist_data, chars_voc = load_historical(f)

        with open(train_config.postrain_file, 'r') as f:
            pos_data, _, pos_voc = load_conll(f, chars_voc=chars_voc)

        train_ds = join_training_data(pos_data, hist_data)

        posval_ds = None
        if train_config.posval_file:
            with open(train_config.posval_file, 'r') as f:
                posval_data, _, _ = load_conll(f, chars_voc=chars_voc, pos_voc=pos_voc)
            posval_ds = IndexableDataset(posval_data)

        histval_ds = None
        if train_config.histval_file:
            with open(train_config.histval_file, 'r') as f:
                histval_data, _ = load_historical(f, chars_voc=chars_voc)
            histval_ds = IndexableDataset(histval_data)

        net_config = HistPOSTaggerConfiguration(alphabet_size=len(chars_voc), pos_dimension=len(pos_voc))
        if args.net_config is not None:
            with open(args.net_config, 'r') as f:
                net_config.load_json(f)

        tagger = HistPOSTagger(net_config)

        print('Training configuration:\n' + train_config.dump_json(), file=sys.stderr)
        print('Network configuration:\n' + net_config.dump_json(), file=sys.stderr)

        save_metadata(args.taggermodel, net_config, chars_voc, pos_voc)
        train(tagger, train_config, train_ds, args.taggermodel,
              pos_validation_set=posval_ds, hist_validation_set=histval_ds)
    elif args.mode == "predict":
        net_config, chars_voc, pos_voc = load_metadata(args.taggermodel)
        tagger = HistPOSTagger(net_config)
        with open(args.test_file, 'r') if args.test_file is not None else sys.stdin as f:
            test_ds = load_vertical(f, chars_voc)
        predict(tagger, test_ds, args.taggermodel, pos_voc, embedder=args.embedder)
    elif args.mode == "eval":
        net_config, chars_voc, pos_voc = load_metadata(args.taggermodel)
        tagger = HistPOSTagger(net_config)
        with open(args.test_file, 'r') if args.test_file is not None else sys.stdin as f:
            test_data, _, _ = load_conll(f, chars_voc=chars_voc, pos_voc=pos_voc)
        test_ds = IndexableDataset(test_data)
        evaluate(tagger, test_ds, args.taggermodel, pos_voc, embedder=args.embedder)


if __name__ == '__main__':
    main()
