from blocks import initialization, main_loop
from blocks.algorithms import StepClipping, GradientDescent, CompositeRule, RMSProp
from blocks.bricks import Linear, NDimensionalSoftmax, Tanh
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import GatedRecurrent, LSTM, SimpleRecurrent
from blocks.bricks.sequence_generators import LookupFeedback, Readout, SequenceGenerator, SoftmaxEmitter
from blocks.bricks.lookup import LookupTable
from blocks.extensions import FinishAfter, Timing, Printing, saveload
from blocks.extensions.training import SharedVariableModifier
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_dropout
from blocks.model import Model
from blocks.monitoring import aggregation

from fuel.datasets.base import IndexableDataset
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from fuel.utils import do_not_pickle_attributes

from theano import tensor


import collections
import itertools
import numpy
import theano


# Most of this file is copied from
# https://github.com/johnarevalo/blocks-char-rnn

# Define this class to skip serialization of extensions
@do_not_pickle_attributes('extensions')
class MainLoop(main_loop.MainLoop):

    def __init__(self, **kwargs):
        super(MainLoop, self).__init__(**kwargs)

    def load(self):
        self.extensions = []


def initialize(to_init):
    for bricks in to_init:
        bricks.weights_init = initialization.Uniform(width=0.08)
        bricks.biases_init = initialization.Constant(0)
        bricks.initialize()


def softmax_layer(h, y, vocab_size, hidden_size):
    hidden_to_output = Linear(name='hidden_to_output', input_dim=hidden_size,
                              output_dim=vocab_size)
    initialize([hidden_to_output])
    linear_output = hidden_to_output.apply(h)
    linear_output.name = 'linear_output'
    softmax = NDimensionalSoftmax()
    y_hat = softmax.apply(linear_output, extra_ndim=1)
    y_hat.name = 'y_hat'
    cost = softmax.categorical_cross_entropy(
        y, linear_output, extra_ndim=1).mean()
    cost.name = 'cost'
    return y_hat, cost


def rnn_layer(dim, h, n):
    linear = Linear(input_dim=dim, output_dim=dim, name='linear' + str(n))
    rnn = SimpleRecurrent(dim=dim, activation=Tanh(), name='rnn' + str(n))
    initialize([linear, rnn])
    return rnn.apply(linear.apply(h))


def gru_layer(dim, h, n):
    fork = Fork(output_names=['linear' + str(n), 'gates' + str(n)],
                name='fork' + str(n), input_dim=dim, output_dims=[dim, dim * 2])
    gru = GatedRecurrent(dim=dim, name='gru' + str(n))
    initialize([fork, gru])
    linear, gates = fork.apply(h)
    return gru.apply(linear, gates)


def lstm_layer(dim, h, n):
    linear = Linear(input_dim=dim, output_dim=dim * 4, name='linear' + str(n))
    lstm = LSTM(dim=dim, name='lstm' + str(n))
    initialize([linear, lstm])
    return lstm.apply(linear.apply(h))


def decoder(h, y, vocab_size, hidden_size, feedback_size, model):
    transition = None
    if model == 'rnn':
        transition = SimpleRecurrent(dim=hidden_size, name='gen_rnn')
    elif model == 'gru':
        transition = GatedRecurrent(dim=hidden_size, name='gen_gru')
    elif model == 'lstm':
        transition = LSTM(dim=hidden_size, name='gen_lstm')

    emitter = SoftmaxEmitter(name='gen_emitter')
    feedback = LookupFeedback(vocab_size, feedback_size, name='gen_feedback')
    readout = Readout(emitter=emitter, feedback_brick=feedback, readout_dim=vocab_size, name='gen_readout')
    generator = SequenceGenerator(readout, transition, name='gen_generator')
    initialize([emitter, feedback, readout, generator])

    batch_cost = generator.cost_matrix(y).sum()
    cost = aggregation.mean(batch_cost, batch_size)


def nn_fprop(x, y, vocab_size, hidden_size, num_layers, model):
    lookup = LookupTable(length=vocab_size, dim=hidden_size)
    initialize([lookup])
    h = lookup.apply(x)
    cells = []
    for i in range(num_layers):
        if model == 'rnn':
            h = rnn_layer(hidden_size, h, i)
        if model == 'gru':
            h = gru_layer(hidden_size, h, i)
        if model == 'lstm':
            h, c = lstm_layer(hidden_size, h, i)
            cells.append(c)
    # return softmax_layer(h, y, vocab_size, hidden_size) + (cells, )
    return decoder(h, y, vocab_size, hidden_size, hidden_size, model)


def load_historical(part, voc=None):
    infile = '/home/stp98/evapet/christian/german.de-hs.%s.hsde' % part
    with open(infile, 'r') as f:
        items = [tuple(line.rstrip('\n').split('\t')) for line in f]

    maxraw = max(len(x[0]) for x in items)
    maxnorm = max(len(x[1]) for x in items)

    raw = numpy.zeros((len(items), maxraw), dtype=numpy.int8)
    norm = numpy.zeros((len(items), maxnorm), dtype=numpy.int8)

    if voc is None:
        chars = set(itertools.chain(*(x[0] + x[1] for x in items)))
        voc = {c: i + 2 for i, c in enumerate(sorted(chars))}
        voc['empty'] = 0
        voc['oov'] = 1

    for i, (r, n) in enumerate(items):
        raw[i, 0:len(r)] = [voc.get(c, voc['oov']) for c in r]
        norm[i, 0:len(n)] = [voc.get(c, voc['oov']) for c in n]

    data = collections.OrderedDict()
    data['raw'] = raw
    data['norm'] = norm
    return IndexableDataset(data), voc


def train():
    nexamples = 1000
    batch_size = 100

    hidden_size = 150
    num_layers = 1
    model = 'gru'

    nepochs = 2
    dropout = 0
    learning_rate = .01
    learning_rate_decay = 0.97
    rmsprop_decay_rate = .95
    step_clipping = 1.0

    trainset, voc = load_historical('train')
    devset, _ = load_historical('dev', voc=voc)

    vocab_size = len(voc)

    train_stream = DataStream(trainset, iteration_scheme=ShuffledScheme(examples=nexamples, batch_size=batch_size))
    dev_stream = DataStream(devset)

    # MODEL
    x = tensor.matrix('raw', dtype='uint32')
    y = tensor.matrix('norm', dtype='uint32')
    y_hat, cost, cells = nn_fprop(x, y, vocab_size, hidden_size, num_layers, model)

    # COST
    cg = ComputationGraph(cost)

    if dropout > 0:
        # Apply dropout only to the non-recurrent inputs (Zaremba et al. 2015)
        inputs = VariableFilter(theano_name_regex=r'.*apply_input.*')(cg.variables)
        cg = apply_dropout(cg, inputs, dropout)
        cost = cg.outputs[0]

    # Learning algorithm
    step_rules = [RMSProp(learning_rate=learning_rate, decay_rate=rmsprop_decay_rate),
                  StepClipping(step_clipping)]
    algorithm = GradientDescent(cost=cost, parameters=cg.parameters,
                                step_rule=CompositeRule(step_rules))

    # Extensions
    gradient_norm = aggregation.mean(algorithm.total_gradient_norm)
    step_norm = aggregation.mean(algorithm.total_step_norm)
    monitored_vars = [cost, gradient_norm, step_norm]

    dev_monitor = DataStreamMonitoring(variables=[cost], after_epoch=True,
                                       before_first_epoch=True, data_stream=dev_stream, prefix="dev")
    train_monitor = TrainingDataMonitoring(variables=monitored_vars, after_batch=True,
                                           before_first_epoch=True, prefix='tra')

    extensions = [dev_monitor, train_monitor, Timing(), Printing(after_batch=True),
                  FinishAfter(after_n_epochs=nepochs),
                  # saveload.Load(load_path),
                  # saveload.Checkpoint(last_path),
                  ]  # + track_best('dev_cost', save_path)

    if learning_rate_decay not in (0, 1):
        extensions.append(SharedVariableModifier(step_rules[0].learning_rate,
                                                 lambda n, lr: numpy.cast[theano.config.floatX](
                                                     learning_rate_decay * lr), after_epoch=True, after_batch=False))

    print('number of parameters in the model: ' + str(tensor.sum([p.size for p in cg.parameters]).eval()))
    # Finally build the main loop and train the model
    mainloop = MainLoop(data_stream=train_stream, algorithm=algorithm,
                        model=Model(cost), extensions=extensions)
    mainloop.run()
