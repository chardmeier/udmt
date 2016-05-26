from blocks.bricks import Tanh
from blocks.bricks.recurrent import Bidirectional, SimpleRecurrent
from blocks.initialization import Orthogonal
from numpy.testing import assert_allclose
from theano import tensor


import itertools
import numpy
import theano
import unittest


class TestBidirectional(unittest.TestCase):
    def setUp(self):
        self.bidir = Bidirectional(weights_init=Orthogonal(),
                                   prototype=SimpleRecurrent(
                                       dim=3, activation=Tanh()))
        self.simple = SimpleRecurrent(dim=3, weights_init=Orthogonal(),
                                      activation=Tanh(), seed=1)
        self.bidir.allocate()
        self.simple.initialize()
        self.bidir.children[0].parameters[0].set_value(
            self.simple.parameters[0].get_value())
        self.bidir.children[1].parameters[0].set_value(
            self.simple.parameters[0].get_value())
        self.x_val = 0.1 * numpy.asarray(
            list(itertools.permutations(range(4))),
            dtype=theano.config.floatX)
        self.x_val = (numpy.ones((24, 4, 3), dtype=theano.config.floatX) *
                      self.x_val[..., None])
        self.mask_val = numpy.ones((24, 4), dtype=theano.config.floatX)
        self.mask_val[12:24, 3] = 0

    def test(self):
        x = tensor.tensor3('x')
        mask = tensor.matrix('mask')
        calc_bidir = theano.function([x, mask],
                                     [self.bidir.apply(x, mask=mask)])
        calc_simple = theano.function([x, mask],
                                      [self.simple.apply(x, mask=mask)])
        h_bidir = calc_bidir(self.x_val, self.mask_val)[0]
        h_simple = calc_simple(self.x_val, self.mask_val)[0]
        h_simple_rev = calc_simple(self.x_val[::-1], self.mask_val[::-1])[0]

        output_names = self.bidir.apply.outputs

        assert output_names == ['states']
        assert_allclose(h_simple, h_bidir[..., :3], rtol=1e-04)
        assert_allclose(h_simple_rev, h_bidir[::-1, ...,  3:], rtol=1e-04)