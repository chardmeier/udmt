from blocks.bricks import application, Initializable, NDimensionalSoftmax, Tanh
from blocks.bricks.parallel import Fork, Merge
from blocks.bricks.recurrent import GatedRecurrent, LSTM, SimpleRecurrent
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal
from blocks.utils import dict_union
from hist.word import BidirectionalWithCombination, CombineExtreme, Encoder
from theano import tensor


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
    def apply(self, chars, char_mask):
        return self.decoder.apply(self.encoder.apply(chars, char_mask))

