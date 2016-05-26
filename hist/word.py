# From blocks-examples/reverse_words

import argparse
import copy
import logging
import pprint
import itertools
import collections
import math
import numpy
import operator
import sys

import theano
from picklable_itertools.extras import equizip
from theano import tensor

from blocks.bricks import Tanh, Initializable
from blocks.bricks.base import application, Brick
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import SimpleRecurrent, Bidirectional, LSTM, GatedRecurrent
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.parallel import Fork
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)
from blocks.config import config
from blocks.graph import ComputationGraph
from fuel.transformers import Mapping, Batch, Padding, Filter
from fuel.datasets import IndexableDataset
from fuel.schemes import ConstantScheme, ShuffledScheme
from blocks.serialization import load_parameters
from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule, RMSProp, Adam, Momentum, AdaGrad)
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.filter import VariableFilter
from blocks.utils import dict_union

from blocks.search import BeamSearch

config.recursion_limit = 100000
floatX = theano.config.floatX
logger = logging.getLogger(__name__)

# Dictionaries
# all_chars = ([chr(ord('a') + i) for i in range(26)] +
#              [chr(ord('0') + i) for i in range(10)] +
#              [',', '.', '!', '?', '<UNK>'] +
#              [' ', '<S>', '</S>'])
# code2char = dict(enumerate(all_chars))
# char2code = {v: k for k, v in code2char.items()}


# def reverse_words(sample):
#     sentence = sample[0]
#     result = []
#     word_start = -1
#     for i, code in enumerate(sentence):
#         if code >= char2code[' ']:
#             if word_start >= 0:
#                 result.extend(sentence[i - 1:word_start - 1:-1])
#                 word_start = -1
#             result.append(code)
#         else:
#             if word_start == -1:
#                 word_start = i
#     return (result,)
#
#
def _lower(s):
    return s.lower()


def _transpose(data):
    return tuple(array.T for array in data)


def _filter_long(data):
    return len(data[0]) <= 100


def _is_nan(log):
    return math.isnan(log.current_row['total_gradient_norm'])


def load_historical(infile, voc=None, autoencoder=False):
    with open(infile, 'r') as f:
        if autoencoder:
            items = [(line.rstrip('\n'),) * 2 for line in f]
        else:
            items = [tuple(line.rstrip('\n').split('\t')) for line in f]

    if voc is None:
        chars = set(itertools.chain(*(x[0] + x[1] for x in items)))
        voc = {c: i + 3 for i, c in enumerate(sorted(chars))}
        voc['<UNK>'] = 0
        voc['<S>'] = 1
        voc['</S>'] = 2

    bos = [voc['<S>']]
    eos = [voc['</S>']]

    raw = []
    norm = []
    for r, n in items:
        raw.append(bos + [voc.get(c, voc['<UNK>']) for c in r] + eos)
        norm.append(bos + [voc.get(c, voc['<UNK>']) for c in n] + eos)

    data = collections.OrderedDict()
    data['raw'] = raw
    data['norm'] = norm
    return IndexableDataset(data), voc


class Concatenate(Brick):
    def __init__(self, dim_a, dim_b, **kwargs):
        self.dim_a = dim_a
        self.dim_b = dim_b
        super().__init__(**kwargs)

    @application
    def apply(self, aa, bb):
        return [tensor.concatenate(a, b) for a, b in equizip(aa, bb)]

    def get_dim(self, name):
        return self.dim_a + self.dim_b


class CombineExtreme(Initializable):
    def __init__(self, operation, **kwargs):
        self.operation = operation
        kwargs.setdefault('children', []).append(operation)
        super().__init__(**kwargs)

    @application
    def apply(self, forward, backward):
        return self.operation.apply(forward[-1], backward[0])


class BidirectionalWithCombination(Bidirectional):
    def __init__(self, prototype, combiner, **kwargs):
        kwargs.setdefault('children', []).append(combiner)
        super().__init__(prototype, **kwargs)

    @application
    def apply(self, *args, **kwargs):
        forward = self.children[0].apply(as_list=True, *args, **kwargs)
        backward = [x[::-1] for x in
                    self.children[1].apply(reverse=True, as_list=True,
                                           *args, **kwargs)]
        return self.combiner.apply(forward, backward)

    def get_dim(self, name):
        return self.combiner.get_dim(name)


class Encoder(Initializable):
    def __init__(self, alphabet_size, dimension, transition, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        forward = copy.deepcopy(transition)
        forward.name = 'forward'
        backward = copy.deepcopy(transition)
        backward.name = 'backward'

        fork = Fork([name for name in transition.apply.sequences
                     if name != 'mask'])
        fork.input_dim = dimension
        fork.output_dims = [transition.get_dim(name) for name in fork.input_names]

        lookup = LookupTable(alphabet_size, dimension)

        self.forward = forward
        self.backward = backward
        self.fork = fork
        self.lookup = lookup
        self.children = [forward, backward, fork, lookup]

    def _push_initialization_config(self):
        super()._push_initialization_config()
        self.forward.weights_init = Orthogonal()
        self.backward.weights_init = Orthogonal()

    @application
    def apply(self, chars, chars_mask):
        forward = self.children[0].apply(as_list=True)
        backward = [x[::-1] for x in
                    self.children[1].apply(reverse=True, as_list=True)]
        return self.encoder.apply(
            **dict_union(
                self.fork.apply(self.lookup.apply(chars), as_dict=True),
                mask=chars_mask))


class Decoder(Initializable):
    def __init__(self, dimension, alphabet_size, transition, **kwargs):
        super(Decoder, self).__init__(**kwargs)

        attention = SequenceContentAttention(
            state_names=transition.apply.states,
            attended_dim=2 * dimension, match_dim=dimension, name="attention")
        readout = Readout(
            readout_dim=alphabet_size,
            source_names=[transition.apply.states[0],
                          attention.take_glimpses.outputs[0]],
            emitter=SoftmaxEmitter(name="emitter"),
            feedback_brick=LookupFeedback(alphabet_size, dimension),
            name="readout")
        generator = SequenceGenerator(
            readout=readout, transition=transition, attention=attention,
            name="generator")

        self.generator = generator
        self.children = [generator]

    def _push_initialization_config(self):
        super()._push_initialization_config()
        self.generator.transition.weights_init = Orthogonal()

    @application
    def cost(self, encoded, encoded_mask, targets, targets_mask):
        return self.generator.cost_matrix(targets, targets_mask, attended=encoded, attended_mask=encoded_mask)

    @application
    def generate(self, encoded, encoded_mask, n_steps, batch_size):
        return self.decoder.generate(
            n_steps=n_steps, batch_size=batch_size,
            attended=encoded, attended_mask=encoded_mask)


class WordTransformer(Initializable):
    """The top brick.

    It is often convenient to gather all bricks of the model under the
    roof of a single top brick.

    """

    def __init__(self, dimension, alphabet_size, recurrent_type='rnn', **kwargs):
        super(WordTransformer, self).__init__(**kwargs)

        if recurrent_type == 'rnn':
            enc_transition = SimpleRecurrent(activation=Tanh(), dim=dimension)
            dec_transition = SimpleRecurrent(activation=Tanh(), dim=dimension)
        elif recurrent_type == 'lstm':
            enc_transition = LSTM(dim=dimension)
            dec_transition = SimpleRecurrent(activation=Tanh(), dim=dimension)
        elif recurrent_type == 'gru':
            enc_transition = GatedRecurrent(dim=dimension)
            dec_transition = SimpleRecurrent(activation=Tanh(), dim=dimension)
        else:
            raise ValueError('Invalid recurrent_type: ' + recurrent_type)

        encoder = Encoder(alphabet_size, dimension, enc_transition)
        decoder = Decoder(dimension, alphabet_size, dec_transition)

        self.encoder = encoder
        self.decoder = decoder
        self.children = [encoder, decoder]

    @application
    def cost(self, chars, chars_mask, targets, targets_mask):
        return self.decoder.cost(
            self.encoder.apply(chars, chars_mask), chars_mask,
            targets, targets_mask)

    @application
    def generate(self, chars):
        chars_mask = tensor.ones(chars.shape)
        return self.decoder.generate(
            n_steps=3 * chars.shape[0], batch_size=chars.shape[1],
            attended=self.encoder.apply(chars, chars_mask),
            attended_mask=chars_mask)


def train(transformer, dataset, num_batches, save_path, step_rule='original'):
    # Data processing pipeline

    dataset.example_iteration_scheme = ShuffledScheme(dataset.num_examples, 10)
    data_stream = dataset.get_example_stream()
    data_stream = Filter(data_stream, _filter_long)
    # data_stream = Batch(data_stream, iteration_scheme=ConstantScheme(10))
    data_stream = Padding(data_stream)
    data_stream = Mapping(data_stream, _transpose)

    # Initialization settings
    transformer.weights_init = IsotropicGaussian(0.1)
    transformer.biases_init = Constant(0.0)
    transformer.push_initialization_config()

    # Build the cost computation graph
    raw = tensor.lmatrix("raw")
    raw_norm = tensor.matrix("raw_mask")
    norm = tensor.lmatrix("norm")
    norm_mask = tensor.matrix("norm_mask")
    batch_cost = transformer.cost(
        raw, raw_norm, norm, norm_mask).sum()
    batch_size = raw.shape[1].copy(name="batch_size")
    cost = aggregation.mean(batch_cost, batch_size)
    cost.name = "sequence_log_likelihood"
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
    generator = transformer.decoder.generator
    (energies,) = VariableFilter(
        applications=[generator.readout.readout],
        name_regex="output")(cg.variables)
    (activations,) = VariableFilter(
        applications=[generator.transition.apply],
        name=generator.transition.apply.states[0])(cg.variables)
    max_length = raw.shape[0].copy(name="max_length")
    cost_per_character = aggregation.mean(
        batch_cost, batch_size * max_length).copy(
        name="character_log_likelihood")
    min_energy = energies.min().copy(name="min_energy")
    max_energy = energies.max().copy(name="max_energy")
    mean_activation = abs(activations).mean().copy(
        name="mean_activation")
    observables = [
         cost, min_energy, max_energy, mean_activation,
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


def predict(transformer, mode, save_path, voc):
    reverse_voc = {idx: word for word, idx in voc.items()}

    raw = tensor.lmatrix("input")
    generated = transformer.generate(raw)
    model = Model(generated)
    logger.info("Loading the model..")
    with open(save_path, 'rb') as f:
        model.set_parameter_values(load_parameters(f))

    sample_expr, = VariableFilter(
        applications=[transformer.generator.generate], name="outputs")(
        ComputationGraph(generated[1]))
    beam_search = BeamSearch(sample_expr)

    def generate(input_):
        """Generate output sequences for an input sequence.

        Incapsulates most of the difference between sampling and beam
        search.

        Returns
        -------
        outputs : list of lists
            Trimmed output sequences.
        costs : list
            The negative log-likelihood of generating the respective
            sequences.

        """
        if mode == "beam_search":
            outputs, costs = beam_search.search(
                {raw: input_}, voc['</S>'],
                3 * input_.shape[0])
        else:
            _1, outputs, _2, _3, costs = (
                model.get_theano_function()(input_))
            outputs = list(outputs.T)
            costs = list(costs.T)
            for i in range(len(outputs)):
                outputs[i] = list(outputs[i])
                try:
                    true_length = outputs[i].index(voc['</S>']) + 1
                except ValueError:
                    true_length = len(outputs[i])
                outputs[i] = outputs[i][:true_length]
                costs[i] = costs[i][:true_length].sum()
        return outputs, costs

    batch_size = 50
    for fline in sys.stdin:
        line, target = tuple(fline.rstrip('\n').split('\t'))

        encoded_input = [voc.get(char, voc["<UNK>"])
                         for char in line.strip()]
        encoded_input = ([voc['<S>']] + encoded_input +
                         [voc['</S>']])
        # print("Encoder input:", encoded_input)
        # print("Target: ", target)

        samples, costs = generate(
            numpy.repeat(numpy.array(encoded_input)[:, None],
                         batch_size, axis=1))

        best = min(zip(costs, samples))
        pred = ''.join(reverse_voc[code] for code in best[1] if code not in {voc['<S>'], voc['</S>']})

        print('%s\t%s\t%s' % (line, target, pred))

        # messages = []
        # for sample, cost in equizip(samples, costs):
        #     message = "({})".format(cost)
        #     message += "".join(reverse_voc[code] for code in sample)
        #     if sample == target:
        #         message += " CORRECT!"
        #     messages.append((cost, message))
        # messages.sort(key=operator.itemgetter(0), reverse=True)
        # for _, message in messages:
        #     print(message)


def main():
    parser = argparse.ArgumentParser(
        "Historical text normaliser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "mode", choices=["train", "sample", "beam_search"],
        help="The mode to run. In the `train` mode a model is trained."
             " In the `sample` and `beam_search` modes a trained model is "
             " to used reverse words in the input text.")
    parser.add_argument(
        "model",
        help="The path to save the training process if the mode"
             " is `train` OR path to an `.tar` files with learned"
             " parameters if the mode is `test`.")
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
        "--autoencoder", dest='autoencoder', action='store_true',
        help="Train an autoencoder instead of a transducer")
    parser.set_defaults(autoencoder=False)
    args = parser.parse_args()

    dataset, voc = load_historical(args.traincorpus, autoencoder=args.autoencoder)

    transformer = WordTransformer(100, len(voc), name="transformer", recurrent_type=args.recurrent_type)

    if args.mode == "train":
        num_batches = args.num_batches
        train(transformer, dataset, num_batches, args.model, step_rule=args.step_rule)
    elif args.mode == "sample" or args.mode == "beam_search":
        predict(transformer, args.mode, args.model, voc)


if __name__ == '__main__':
    main()
