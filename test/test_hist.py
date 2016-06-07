from hist.word import BidirectionalWithCombination, CombineExtreme, Concatenate, _collect_mask
from blocks.bricks import Tanh
from blocks.bricks.parallel import Merge
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.initialization import Constant, Orthogonal
from numpy.testing import assert_allclose, assert_equal
from theano import tensor

import itertools
import numpy
import theano
import unittest


class TestBidirectionalWithCombination(unittest.TestCase):
    def setUp(self):
        self.simple = SimpleRecurrent(dim=3, weights_init=Orthogonal(),
                                      activation=Tanh(), seed=1)
        self.simple.initialize()
        self.x_val = 0.1 * numpy.asarray(
            list(itertools.permutations(range(4))),
            dtype=theano.config.floatX)
        self.x_val = (numpy.ones((24, 4, 3), dtype=theano.config.floatX) *
                      self.x_val[..., None])
        self.mask_val = numpy.ones((24, 4), dtype=theano.config.floatX)
        self.mask_val[12:24, 3] = 0

    def _setup_bidir(self, combiner):
        bidir = BidirectionalWithCombination(weights_init=Orthogonal(),
                                             combiner=combiner,
                                             prototype=SimpleRecurrent(
                                                 dim=3, activation=Tanh()))
        bidir.allocate()
        bidir.children[0].parameters[0].set_value(self.simple.parameters[0].get_value())
        bidir.children[1].parameters[0].set_value(self.simple.parameters[0].get_value())
        return bidir

    def test_concatenation(self):
        bidir = self._setup_bidir(Concatenate(input_dims=[3, 3], input_names=['forward', 'backward']))
        x = tensor.tensor3('x')
        mask = tensor.matrix('mask')
        calc_bidir = theano.function([x, mask],
                                     [bidir.apply(x, mask=mask)])
        calc_simple = theano.function([x, mask],
                                      [self.simple.apply(x, mask=mask)])
        h_bidir = calc_bidir(self.x_val, self.mask_val)[0]
        h_simple = calc_simple(self.x_val, self.mask_val)[0]
        h_simple_rev = calc_simple(self.x_val[::-1], self.mask_val[::-1])[0]

        output_names = bidir.apply.outputs

        assert output_names == ['states']
        assert_allclose(h_simple, h_bidir[..., :3], rtol=1e-04)
        assert_allclose(h_simple_rev, h_bidir[::-1, ...,  3:], rtol=1e-04)

    def test_merge(self):
        # merge all time steps individually...
        operation = Merge(input_names=['forward', 'backward'], input_dims=[3, 3],
                          output_dim=4, weights_init=Orthogonal(),
                          biases_init=Constant(1.8))
        operation.children[0].use_bias = True
        operation.push_initialization_config()
        bidir = self._setup_bidir(operation)
        operation.initialize()

        x = tensor.tensor3('x')
        mask = tensor.matrix('mask')
        calc_bidir = theano.function([x, mask],
                                     [bidir.apply(x, mask=mask)])
        calc_simple = theano.function([x, mask],
                                      [self.simple.apply(x, mask=mask)])
        h_bidir = calc_bidir(self.x_val, self.mask_val)[0]
        h_simple = calc_simple(self.x_val, self.mask_val)[0]
        h_simple_rev = calc_simple(self.x_val[::-1], self.mask_val[::-1])[0]

        w_f = operation.children[0].parameters[0].get_value()
        w_b = operation.children[1].parameters[0].get_value()
        bias = operation.children[0].parameters[1].get_value()

        c_simple = numpy.dot(h_simple, w_f) + \
                   numpy.dot(h_simple_rev[::-1], w_b) + bias

        output_names = bidir.apply.outputs

        assert output_names == ['states']
        assert_allclose(c_simple, h_bidir, rtol=1e-04)

    def test_combine_extreme_concat(self):
        operation = Concatenate([3, 3], input_names=['forward', 'backward'])
        bidir = self._setup_bidir(CombineExtreme(operation))

        x = tensor.tensor3('x')
        mask = tensor.matrix('mask')
        calc_bidir = theano.function([x, mask],
                                     [bidir.apply(x, mask=mask)])
        calc_simple = theano.function([x, mask],
                                      [self.simple.apply(x, mask=mask)])
        h_bidir = calc_bidir(self.x_val, self.mask_val)[0]
        h_simple = calc_simple(self.x_val, self.mask_val)[0]
        h_simple_rev = calc_simple(self.x_val[::-1], self.mask_val[::-1])[0]

        fwd_last = h_simple[-1, ...]
        bwd_first = h_simple_rev[-1, ...]

        c_simple = numpy.concatenate([fwd_last, bwd_first], axis=-1)

        output_names = bidir.apply.outputs

        assert output_names == ['states']
        assert_allclose(c_simple, h_bidir, rtol=1e-04)

    def test_combine_extreme_merge(self):
        operation = Merge(input_names=['forward', 'backward'], input_dims=[3, 3],
                          output_dim=3, weights_init=Orthogonal(),
                          biases_init=Constant(1.8))
        operation.children[0].use_bias = True
        operation.push_initialization_config()
        bidir = self._setup_bidir(CombineExtreme(operation))
        operation.initialize()

        x = tensor.tensor3('x')
        mask = tensor.matrix('mask')
        calc_bidir = theano.function([x, mask],
                                     [bidir.apply(x, mask=mask)])
        calc_simple = theano.function([x, mask],
                                      [self.simple.apply(x, mask=mask)])
        h_bidir = calc_bidir(self.x_val, self.mask_val)[0]
        h_simple = calc_simple(self.x_val, self.mask_val)[0]
        h_simple_rev = calc_simple(self.x_val[::-1], self.mask_val[::-1])[0]

        w_f = operation.children[0].parameters[0].get_value()
        w_b = operation.children[1].parameters[0].get_value()
        bias = operation.children[0].parameters[1].get_value()

        c_simple = numpy.dot(h_simple[-1, ...], w_f) + \
                   numpy.dot(h_simple_rev[-1, ...], w_b) + bias

        output_names = bidir.apply.outputs

        assert output_names == ['states']
        assert_allclose(c_simple, h_bidir, rtol=1e-04)


class TestCombineWords(unittest.TestCase):
    def test_collect_mask(self):
        t_data = tensor.tensor3()
        t_mask = tensor.imatrix()

        collect_fn = theano.function([t_data, t_mask], list(_collect_mask(t_data, t_mask)), on_unused_input='warn')

        data = numpy.random.randn(10, 2, 7)
        # data = numpy.arange(20).reshape((10, 2, 1))
        mask = numpy.transpose(numpy.array([[1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                                            [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]], dtype=numpy.int8))

        output, outmask = collect_fn(data, mask)

        expected = numpy.zeros_like(data)
        expected[(0, 1, 2, 3, 0, 1, 2, 3, 4), (0, 0, 0, 0, 1, 1, 1, 1, 1), :] =\
            data[(0, 3, 5, 9, 0, 2, 5, 7, 9), (0, 0, 0, 0, 1, 1, 1, 1, 1), :]

        expected_mask = numpy.transpose(numpy.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                                     [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=numpy.int8))

        assert_allclose(output, expected, rtol=1e-4)
        assert_equal(outmask, expected_mask)


if __name__ == '__main__':
    unittest.main()