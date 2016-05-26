from blocks.bricks import Tanh, Initializable, NDimensionalSoftmax
from blocks.bricks.recurrent import Bidirectional


class TagDecoder(Initializable):
    def __init__(self, transition, **kwargs):
        super().__init__(**kwargs)

        sequence = Bidirectional(transition, name='pos_sequence')
        softmax = NDimensionalSoftmax(name='pos_softmax')

        self.sequence = sequence
        self.softmax = softmax
        self.children = [sequence, softmax]

    @application
    def cost(self, word_enc, word_enc_mask, targets, targets_mask):
        return self.softmax.categorical_cross_entropy(targets, word_enc, extra_ndim=1)