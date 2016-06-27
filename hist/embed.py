from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule, RMSProp, Adam, Momentum, AdaGrad)
from blocks.bricks import application, Initializable, Logistic, MLP, Tanh
from blocks.config import config
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.serialization import load_parameters
from hist.word import CombineWords, Encoder
from hist.tagger import make_merge_combiner
from fuel.transformers import Batch, FilterSources, Mapping, Padding
from fuel.datasets import IndexableDataset
from fuel.schemes import ConstantScheme, ShuffledScheme
from theano import tensor

import argparse
import collections
import fuel
import itertools
import logging
import math
import numpy
import pprint
import theano


config.recursion_limit = 100000
floatX = theano.config.floatX
logger = logging.getLogger(__name__)


def load_training(infile, chars_voc=None):
    floatx_vals = numpy.arange(2, dtype=fuel.config.floatX)
    zero = floatx_vals[0]
    one = floatx_vals[1]

    seqs = []
    for key, group in itertools.groupby(infile, lambda l: l == '\n'):
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
    all_prediction_truth = []
    for snt_pairs in seqs:
        hist_chars = [chars_voc['<S>']]
        norm_chars = [chars_voc['<S>']]
        hist_word_mask = [one]
        norm_word_mask = [one]
        prediction_truth = numpy.zeros((len(snt_pairs), len(chars_voc)), dtype=fuel.config.floatX)
        for i, (hist, norm) in enumerate(snt_pairs):
            hist_word = list(chars_voc.get(c, chars_voc['<UNK>']) for c in hist)
            hist_chars.extend(hist_word)
            hist_chars.append(chars_voc[' '])
            hist_word_mask.extend([zero] * len(hist) + [one])
            norm_word = list(chars_voc.get(c, chars_voc['<UNK>']) for c in norm)
            norm_chars.extend(norm_word)
            norm_chars.append(chars_voc[' '])
            norm_word_mask.extend([zero] * len(norm) + [one])
            prediction_truth[i, norm_word] = one
        hist_chars.append(chars_voc['</S>'])
        norm_chars.append(chars_voc['</S>'])
        hist_word_mask.append(one)
        norm_word_mask.append(one)

        all_hist.append(hist_chars)
        all_norm.append(norm_chars)
        all_hist_word_mask.append(hist_word_mask)
        all_norm_word_mask.append(norm_word_mask)
        all_prediction_truth.append(prediction_truth)

    data = collections.OrderedDict()
    data['hist'] = all_hist
    data['hist_word_mask'] = all_hist_word_mask
    data['norm'] = all_norm
    data['norm_word_mask'] = all_norm_word_mask
    data['prediction_truth'] = all_prediction_truth

    return IndexableDataset(data), chars_voc


class ContextEmbedder(Initializable):
    def __init__(self, alphabet_size, dimension, hidden_size, transition,
                 prediction_weight, word_boundary, **kwargs):
        super().__init__(**kwargs)

        combiner = CombineWords(make_merge_combiner(dimension))
        encoder = Encoder(alphabet_size, dimension, transition, combiner)

        predictor = MLP([Tanh(), Logistic()], dims=[dimension, hidden_size, alphabet_size])

        self.prediction_weight = prediction_weight
        self.word_boundary = word_boundary

        self.encoder = encoder
        self.predictor = predictor

        self.children = [encoder, predictor]

    @application
    def cost(self, hist, hist_mask, norm, norm_mask, prediction_truth):
        hist_enc, collected_mask = self.encoder.apply(hist, hist_mask)
        norm_enc, _ = self.encoder.apply(norm, norm_mask)

        # Axes: time x batch x features
        diff_cost = (collected_mask * tensor.sqr(norm_enc - hist_enc).sum(axis=2)).sum(axis=0).mean()

        hist_predictions = self.predictor.apply(hist_enc)
        norm_predictions = self.predictor.apply(norm_enc)

        if self.prediction_weight > 0:
            hist_pred_cost = (-(prediction_truth * tensor.log(hist_predictions)).sum(axis=2) * collected_mask).\
                sum(axis=0).mean()
            norm_pred_cost = (-(prediction_truth * tensor.log(norm_predictions)).sum(axis=2) * collected_mask). \
                sum(axis=0).mean()
            pred_cost = 0.5 * hist_pred_cost + 0.5 * norm_pred_cost
        else:
            pred_cost = theano.shared(0)

        return diff_cost + self.prediction_weight * pred_cost

    @application
    def embed(self, chars, chars_mask):
        return self.encoder.apply(chars, chars_mask)


def _transpose(data):
    return tuple(array.T for array in data)


def _is_nan(log):
    return math.isnan(log.current_row['total_gradient_norm'])


def train(embedder, dataset, num_batches, save_path, step_rule='original'):
    # Data processing pipeline

    dataset.example_iteration_scheme = ShuffledScheme(dataset.num_examples, 10)
    data_stream = dataset.get_example_stream()
    # data_stream = Filter(data_stream, _filter_long)
    # data_stream = Batch(data_stream, iteration_scheme=ConstantScheme(10))
    data_stream = Padding(data_stream, mask_sources=['chars', 'word_mask', 'pos'])
    data_stream = FilterSources(data_stream, sources=['chars', 'chars_mask', 'word_mask', 'pos'])
    data_stream = Mapping(data_stream, _transpose)

    # Initialization settings
    embedder.weights_init = IsotropicGaussian(0.1)
    embedder.biases_init = Constant(0.0)
    embedder.push_initialization_config()

    # Build the cost computation graph
    hist = tensor.lmatrix("chars")
    hist_mask = tensor.matrix("hist_mask")
    norm = tensor.lmatrix("norm")
    norm_mask = tensor.matrix("norm_mask")
    prediction_truth = tensor.matrix("prediction_truth")
    batch_cost = embedder.cost(hist, hist_mask, norm, norm_mask, prediction_truth).sum()
    batch_size = hist.shape[1].copy(name="batch_size")
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
    max_length = hist.shape[0].copy(name="max_length")
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


def main():
    parser = argparse.ArgumentParser(
        "Create word embeddings for historical text",
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
        dataset, chars_voc = load_training(f)

    tagger = ContextEmbedder(len(chars_voc), 100, 101, args.recurrent_type, 1.0, chars_voc[' '], name="tagger")

    if args.mode == "train":
        num_batches = args.num_batches
        train(tagger, dataset, num_batches, args.model, step_rule=args.step_rule)
    else:
        print("Mode %s not implemented." % args.mode)
    # elif args.mode == "predict":
    #     with open(args.test_file, 'r') if args.test_file is not None else sys.stdin as f:
    #         testset = load_vertical(f, chars_voc)
    #     predict(tagger, testset, args.model, pos_voc)
    # elif args.mode == "eval":
    #     with open(args.test_file, 'r') if args.test_file is not None else sys.stdin as f:
    #         testset, _, _ = load_conll(f, chars_voc=chars_voc, pos_voc=pos_voc)
    #     evaluate(tagger, testset, args.model, pos_voc)


if __name__ == '__main__':
    main()
