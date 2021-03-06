from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule, RMSProp, Adam, Momentum, AdaGrad)
from blocks.bricks import application, Initializable, MLP, NDimensionalSoftmax, Tanh
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Fork, Merge
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
from blocks.utils import dict_union
from dependency import conll_trees
from fuel.transformers import Batch, FilterSources, Mapping, Padding
from fuel.datasets import IndexableDataset
from fuel.schemes import ConstantScheme, ShuffledScheme
from hist.word import BidirectionalWithCombination, CombineWords, Concatenate
from theano import tensor

import argparse
import collections
import copy
import fuel
import itertools
import logging
import math
import numpy
import pprint
import sys
import theano


logger = logging.getLogger(__name__)


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
    data['words'] = words
    data['chars'] = chars
    data['word_mask'] = word_mask
    data['pos'] = pos

    return IndexableDataset(data), chars_voc, pos_voc


def load_vertical(infile, chars_voc):
    floatx_vals = numpy.arange(2, dtype=fuel.config.floatX)
    zero = floatx_vals[0]
    one = floatx_vals[1]

    words = []
    chars = []
    word_mask = []
    for key, group in itertools.groupby(infile, lambda l: l == '\n'):
        if not key:
            snt_words = [w.rstrip('\n') for w in group]
            snt_chars = [chars_voc['<S>']]
            snt_word_mask = [one]
            for word in snt_words:
                snt_chars.extend(chars_voc.get(c, chars_voc['<UNK>']) for c in word)
                snt_chars.append(chars_voc[' '])
                snt_word_mask.extend([zero] * len(word) + [one])
            snt_chars.append(chars_voc['</S>'])
            snt_word_mask.append(one)

            words.append(snt_words)
            chars.append(snt_chars)
            word_mask.append(snt_word_mask)

    data = collections.OrderedDict()
    data['words'] = words
    data['chars'] = chars
    data['word_mask'] = word_mask

    return IndexableDataset(data)


class BidirectionalLayer(Initializable):
    def __init__(self, dimension, prototype, combiner, **kwargs):
        super().__init__(**kwargs)
        sequence = BidirectionalWithCombination(prototype=prototype, combiner=combiner)
        fork = Fork([name for name in prototype.apply.sequences
                     if name != 'mask'])
        fork.input_dim = dimension
        fork.output_dims = [prototype.get_dim(name) for name in fork.input_names]

        self.sequence = sequence
        self.fork = fork
        self.children = [sequence, fork]

    @application
    def apply(self, input_, **kwargs):
        output = self.sequence.apply(
            **dict_union(
                self.fork.apply(input_, as_dict=True),
                kwargs), as_dict=True)
        return output['states']

    @apply.delegate
    def apply_delegate(self):
        return self.sequence


class WordEmbedding(Initializable):
    def __init__(self, alphabet_size, dimensions, transition_type, **kwargs):
        super().__init__(**kwargs)

        if transition_type == 'rnn':
            transitions = [SimpleRecurrent(activation=Tanh(), dim=dim)
                           for dim in dimensions[1:]]
        elif transition_type == 'lstm':
            transitions = [LSTM(dim=dim)
                           for dim in dimensions[1:]]
        elif transition_type == 'gru':
            transitions = [GatedRecurrent(dim=dim)
                           for dim in dimensions[1:]]
        else:
            raise ValueError('Unknown transition type: ' + transition_type)

        lookup = LookupTable(alphabet_size, dimensions[0])
        sequences = []
        for i, (input_dim, output_dim, transition) in enumerate(zip(dimensions, dimensions[1:], transitions[:-1])):
            sequences.append(BidirectionalLayer(input_dim if i == 0 else 2 * input_dim,
                                                weights_init=Orthogonal(),
                                                prototype=transition,
                                                combiner=Concatenate(input_names=['forward', 'backward'],
                                                                     input_dims=[output_dim] * 2),
                                                name='sequence%d' % i))

        final_sequence = BidirectionalLayer(dimensions[-2] if len(dimensions) == 2 else 2 * dimensions[-2],
                                            weights_init=Orthogonal(),
                                            prototype=transitions[-1],
                                            combiner=CombineWords(Concatenate(input_names=['forward', 'backward'],
                                                                              input_dims=[dimensions[-1]] * 2)),
                                            name='sequence_final')

        self.lookup = lookup
        self.sequences = sequences
        self.final_sequence = final_sequence
        self.children = sequences + [final_sequence, lookup]

    @application(inputs=['chars', 'chars_mask', 'word_mask'], outputs=['output', 'mask'])
    def apply(self, chars, chars_mask, word_mask):
        state = self.lookup.apply(chars)
        for seq in self.sequences:
            state = seq.apply(state, mask=chars_mask)
        return self.final_sequence.apply(state, mask=word_mask)


class TagPredictor(Initializable):
    def __init__(self, dimensions, prototype=Tanh(), **kwargs):
        super().__init__(**kwargs)

        activations = [copy.deepcopy(prototype) for _ in dimensions[1:-1]] + [None]
        mlp = MLP(activations=activations, dims=dimensions)
        softmax = NDimensionalSoftmax()

        self.mlp = mlp
        self.softmax = softmax
        self.children = [mlp, softmax]

    @application
    def cost(self, word_enc, mask, targets):
        predictions = self.apply(word_enc)
        widx, exmpl = tensor.nonzero(mask)
        tgtidx = targets[widx, exmpl]
        crossentropy = -tensor.sum(tensor.log(predictions[widx, exmpl, tgtidx])) / word_enc.shape[1]
        return crossentropy

    @application
    def apply(self, word_enc):
        return self.softmax.apply(self.mlp.apply(word_enc), extra_ndim=1)


class POSTagger(Initializable):
    def __init__(self, alphabet_size, char_dimension, word_dimension, pos_dimension,
                 transition_type, **kwargs):
        super().__init__(**kwargs)

        embedder = WordEmbedding(alphabet_size, [char_dimension, char_dimension, word_dimension], transition_type)
        predictor = TagPredictor([word_dimension, pos_dimension])

        self.embedder = embedder
        self.predictor = predictor
        self.children = [embedder, predictor]

    @application
    def cost(self, chars, chars_mask, word_mask, targets):
        encoded, collected_mask = self.embedder.apply(chars, chars_mask, word_mask)
        return self.predictor.cost(encoded, collected_mask, targets)

    @application
    def apply(self, chars, chars_mask, word_mask):
        encoded, collected_mask = self.embedder.apply(chars, chars_mask, word_mask)
        return self.predictor.apply(encoded)


def _transpose(data):
    return tuple(array.T for array in data)


def _is_nan(log):
    return math.isnan(log.current_row['total_gradient_norm'])


def train(postagger, dataset, num_batches, save_path, step_rule='original'):
    # Data processing pipeline

    dataset.example_iteration_scheme = ShuffledScheme(dataset.num_examples, 10)
    data_stream = dataset.get_example_stream()
    # data_stream = Filter(data_stream, _filter_long)
    # data_stream = Batch(data_stream, iteration_scheme=ConstantScheme(10))
    data_stream = Padding(data_stream, mask_sources=('chars', 'word_mask', 'pos'))
    data_stream = FilterSources(data_stream, sources=('chars', 'chars_mask', 'word_mask', 'pos'))
    data_stream = Mapping(data_stream, _transpose)

    # Initialization settings
    postagger.weights_init = IsotropicGaussian(0.1)
    postagger.biases_init = Constant(0.0)
    postagger.push_initialization_config()

    # Build the cost computation graph
    chars = tensor.lmatrix("chars")
    chars_mask = tensor.matrix("chars_mask")
    pos = tensor.lmatrix("pos")
    word_mask = tensor.matrix("word_mask")
    batch_cost = postagger.cost(chars, chars_mask, word_mask, pos).sum()
    batch_size = chars.shape[1].copy(name="batch_size")
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
    max_length = chars.shape[0].copy(name="max_length")
    cost_per_character = aggregation.mean(
        batch_cost, batch_size * max_length).copy(
        name="character_log_likelihood")
    observables = [
         cost,
         batch_size, max_length, cost_per_character,
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
    data_stream = FilterSources(data_stream, sources=('words', 'chars', 'word_mask'))
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
    data_stream = Padding(data_stream, mask_sources=['chars', 'word_mask', 'pos'])
    data_stream = FilterSources(data_stream, sources=('chars', 'chars_mask', 'word_mask', 'pos'))
    data_stream = Mapping(data_stream, _transpose)

    chars = tensor.lmatrix('chars')
    chars_mask = tensor.matrix('chars_mask')
    word_mask = tensor.matrix('word_mask')
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

    print('Accuracy: %5d/%5d = %6f\n' % (matches, total, matches/total))
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
        "model",
        help="The path to save the training process if the mode"
             " is `train` OR path to an `.tar` files with learned"
             " parameters if the mode is `predict`.")
    parser.add_argument(
        "traincorpus",
        help="the training corpus (required for both training and prediction)")
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

    with open(args.traincorpus, 'r') as f:
        dataset, chars_voc, pos_voc = load_conll(f)

    tagger = POSTagger(len(chars_voc), 100, 101, len(pos_voc), args.recurrent_type, name="tagger")

    if args.mode == "train":
        num_batches = args.num_batches
        train(tagger, dataset, num_batches, args.model, step_rule=args.step_rule)
    elif args.mode == "predict":
        with open(args.test_file, 'r') if args.test_file is not None else sys.stdin as f:
            testset = load_vertical(f, chars_voc)
        predict(tagger, testset, args.model, pos_voc)
    elif args.mode == "eval":
        with open(args.test_file, 'r') if args.test_file is not None else sys.stdin as f:
            testset, _, _ = load_conll(f, chars_voc=chars_voc, pos_voc=pos_voc)
        evaluate(tagger, testset, args.model, pos_voc)


if __name__ == '__main__':
    main()
