from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule, RMSProp, Adam, Momentum, AdaGrad)
from blocks.bricks import application, Initializable, NDimensionalSoftmax, Tanh
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
from fuel.transformers import Batch, Mapping, Padding
from fuel.datasets import IndexableDataset
from fuel.schemes import ConstantScheme, ShuffledScheme
from hist.word import BidirectionalWithCombination, CombineExtreme, Encoder
from theano import tensor

import argparse
import collections
import itertools
import logging
import math
import pprint
import sys
import theano


logger = logging.getLogger(__name__)


def load_conll(infile, chars_voc=None, pos_voc=None):
    seqs = []
    with open(infile, 'r') as f:
        for tr in conll_trees(f):
            seqs.append([(n.token, n.pos) for n in tr[1:]])

    if chars_voc is None:
        chars = set(itertools.chain(t[0] for t in itertools.chain(seqs)))
        chars_voc = {c: i + 4 for i, c in enumerate(sorted(chars))}
        chars_voc['<UNK>'] = 0
        chars_voc['<S>'] = 1
        chars_voc['</S>'] = 2
        chars_voc[' '] = 3

    if pos_voc is None:
        tags = set(t[1] for t in itertools.chain(seqs))
        pos_voc = {c: i + 3 for i, c in enumerate(sorted(tags))}
        pos_voc['<UNK>'] = 0
        pos_voc['<S>'] = 1
        pos_voc['</S>'] = 2

    chars = []
    pos = []
    word_mask = []
    for snt in seqs:
        snt_chars = [chars_voc['<S>']]
        snt_pos = [pos_voc['<S>']]
        snt_word_mask = [1]
        for word, pos in snt:
            snt_chars.extend(chars_voc.get(c, chars_voc['<UNK>']) for c in word)
            snt_chars.append(chars_voc[' '])
            snt_pos.extend([0] * len(word) + [pos_voc.get(pos, pos_voc['<UNK>'])])
            snt_word_mask.extend([0] * len(word) + [1])
        snt_chars.append(chars_voc['</S>'])
        snt_pos.append(pos_voc['</S>'])
        snt_word_mask.append(1)

        chars.append(snt_chars)
        pos.append(snt_pos)
        word_mask.append(snt_word_mask)

    data = collections.OrderedDict()
    data['chars'] = chars
    data['pos'] = pos
    data['word_mask'] = word_mask

    return IndexableDataset(data), chars_voc, pos_voc


def load_vertical(infile, chars_voc):
    chars = []
    word_mask = []
    for key, group in itertools.groupby(infile, lambda l: l == '\n'):
        if not key:
            snt_chars = [chars_voc['<S>']]
            snt_word_mask = [1]
            for word in group:
                snt_chars.extend(chars_voc.get(c, chars_voc['<UNK>']) for c in word)
                snt_chars.append(chars_voc[' '])
                snt_word_mask.extend([0] * len(word) + [1])
            snt_chars.append(chars_voc['</S>'])
            snt_word_mask.append(1)

            chars.append(snt_chars)
            word_mask.append(snt_word_mask)

    data = collections.OrderedDict()
    data['chars'] = chars
    data['word_mask'] = word_mask

    return IndexableDataset(data)


class TagDecoder(Initializable):
    def __init__(self, dimension, transition, combiner, **kwargs):
        super().__init__(**kwargs)

        sequence = BidirectionalWithCombination(weights_init=Orthogonal(),
                                                prototype=transition,
                                                combiner=CombineExtreme(combiner),
                                                name='pos_sequence')
        fork = Fork([name for name in sequence.apply.sequences if name != 'mask'])
        fork.input_dim = dimension
        fork.output_dims = [sequence.get_dim(name) for name in fork.input_names]

        softmax = NDimensionalSoftmax(name='pos_softmax')

        self.fork = fork
        self.sequence = sequence
        self.softmax = softmax
        self.children = [sequence, fork, softmax]

    @application
    def cost(self, word_enc, targets, mask=None):
        if mask is None:
            mask = tensor.ones(targets.shape[-1])
        context_enc = self.sequence.apply(
            **dict_union(self.fork.apply(word_enc, as_dict=True), mask=mask))
        return tensor.dot(self.softmax.categorical_cross_entropy(targets, context_enc, extra_ndim=1),
                          mask)

    @application
    def apply(self, word_enc, mask=None):
        context_enc = self.sequence.apply(
            **dict_union(self.fork.apply(word_enc, as_dict=True), mask=mask))
        return self.softmax.apply(context_enc)


def _make_merge_combiner(dimension):
    combiner = Merge(input_names=['forward', 'backward'], input_dims=[dimension] * 2,
                     output_dim=dimension, weights_init=IsotropicGaussian(0.1),
                     biases_init=Constant(0))
    combiner.children[0].use_bias = True
    combiner.push_initialization_config()
    return combiner


class POSTagger(Initializable):
    def __init__(self, alphabet_size, char_dimension, word_dimension,
                 transition_type, **kwargs):
        super().__init__(**kwargs)

        recurrent_dims = [char_dimension, word_dimension]

        if transition_type == 'rnn':
            transitions = [SimpleRecurrent(activation=Tanh(), dim=dim)
                           for dim in recurrent_dims]
        elif transition_type == 'lstm':
            transitions = [LSTM(dim=dim)
                           for dim in recurrent_dims]
        elif transition_type == 'gru':
            transitions = [GatedRecurrent(dim=dim)
                           for dim in recurrent_dims]
        else:
            raise ValueError('Unknown transition type: ' + transition_type)

        encoder = Encoder(alphabet_size, char_dimension, transitions[0], _make_merge_combiner(char_dimension))
        decoder = TagDecoder(word_dimension, transitions[1], _make_merge_combiner(word_dimension))

        self.encoder = encoder
        self.decoder = decoder
        self.children = [encoder, decoder]

    @application
    def cost(self, chars, char_mask, targets, word_mask):
        return self.decoder.cost(self.encoder.apply(chars, char_mask),
                                 targets, mask=word_mask)

    @application
    def apply(self, chars, char_mask, word_mask):
        return self.decoder.apply(self.encoder.apply(chars, char_mask), mask=word_mask)


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
    data_stream = Padding(data_stream)
    data_stream = Mapping(data_stream, _transpose)

    # Initialization settings
    postagger.weights_init = IsotropicGaussian(0.1)
    postagger.biases_init = Constant(0.0)
    postagger.push_initialization_config()

    # Build the cost computation graph
    chars = tensor.lmatrix("chars")
    chars_mask = tensor.matrix("chars_mask")
    pos = tensor.lmatrix("pos")
    pos_mask = tensor.matrix("pos_mask")
    batch_cost = postagger.cost(chars, chars_mask, pos, pos_mask).sum()
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
        observables, prefix="average", every_n_batches=10)

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
            Printing(every_n_batches=1)])
    main_loop.run()


def predict(postagger, dataset, save_path, chars_voc, pos_voc):
    data_stream = dataset.get_example_stream()
    data_stream = Batch(data_stream, iteration_scheme=ConstantScheme(50))
    data_stream = Padding(data_stream)
    data_stream = Mapping(data_stream, _transpose)

    chars = tensor.lmatrix('chars')
    chars_mask = tensor.matrix('chars_mask')
    word_mask = tensor.matrix('word_mask')
    pos = postagger.apply(chars, chars_mask, word_mask)

    model = Model(pos)
    with open(save_path, 'rb') as f:
        model.set_parameter_values(load_parameters(f))

    tag_fn = theano.function(inputs=[chars, chars_mask, word_mask], outputs=pos)

    reverse_chars = {idx: word for word, idx in chars_voc.items()}
    reverse_pos = {idx: word for word, idx in pos_voc.items()}

    for i_chars, i_chars_mask, i_word_mask in data_stream.get_epoch_iterator():
        o_pos = tag_fn(i_chars, i_chars_mask, i_word_mask)
        for i in range(o_pos.shape[0]):
            for j in range(o_pos.shape[1]):
                if not i_word_mask[i, j]:
                    sys.stdout.write(reverse_chars[i_chars[i, j]])
                else:
                    sys.stdout.write('\t%s\n' % reverse_pos[o_pos[i, j]])
            sys.stdout.write('\n')


def main():
    parser = argparse.ArgumentParser(
        "POS tagger",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "mode", choices=["train", "predict"],
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
    parser.set_defaults(autoencoder=False)
    args = parser.parse_args()

    dataset, chars_voc, pos_voc = load_conll(args.traincorpus)

    tagger = POSTagger(len(chars_voc), 100, 100, args.recurrent_type, name="tagger")

    if args.mode == "train":
        num_batches = args.num_batches
        train(tagger, dataset, num_batches, args.model, step_rule=args.step_rule)
    elif args.mode == "predict":
        testset = load_vertical(sys.stdin)
        predict(tagger, testset, args.model, chars_voc, pos_voc)


if __name__ == '__main__':
    main()
