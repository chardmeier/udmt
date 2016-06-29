from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule, RMSProp, Adam, Momentum, AdaGrad)
from blocks.bricks import Tanh
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
from dependency import conll_trees
from fuel.transformers import Batch, FilterSources, Mapping, Padding
from fuel.datasets import IndexableDataset
from fuel.schemes import ConstantScheme, ShuffledScheme
from hist.embed import embed, load_training, text_to_dataset, ContextEmbedder
from hist.tagger import PreembeddedPOSTagger
from theano import tensor

import argparse
import collections
import itertools
import logging
import math
import numpy
import pprint
import sys
import theano


logger = logging.getLogger(__name__)


def _transpose(data):
    return tuple(array.T if array.ndim == 2 else array.transpose(1, 0, 2) for array in data)


def _is_nan(log):
    return math.isnan(log.current_row['total_gradient_norm'])


def load_conll(infile, chars_voc, pos_voc=None):
    tokens = []
    pos = []
    for tr in conll_trees(infile):
        tokens.append([n.token for n in tr[1:]])
        pos.append([n.pos for n in tr[1:]])

    if pos_voc is None:
        tags = set(t for t in itertools.chain(*pos))
        pos_voc = {c: i + 3 for i, c in enumerate(sorted(tags))}
        pos_voc['<UNK>'] = 0
        pos_voc['<S>'] = 1
        pos_voc['</S>'] = 2

    text_ds = text_to_dataset(tokens, chars_voc)
    pos_enc = [[pos_voc['<S>']] + [pos_voc.get(t, pos_voc['<UNK>']) for t in snt] + [pos_voc['</S>']] for snt in pos]

    return text_ds, pos_enc, pos_voc


def load_vertical(infile, chars_voc):
    tokens = []
    for key, group in itertools.groupby(infile, lambda l: l == '\n'):
        if not key:
            tokens.append([w.rstrip('\n') for w in group])

    return text_to_dataset(tokens, chars_voc)


def train(postagger, dataset, num_batches, save_path, step_rule='original'):
    # Data processing pipeline

    dataset.example_iteration_scheme = ShuffledScheme(dataset.num_examples, 10)
    data_stream = dataset.get_example_stream()
    # data_stream = Filter(data_stream, _filter_long)
    # data_stream = Batch(data_stream, iteration_scheme=ConstantScheme(10))
    data_stream = Padding(data_stream)
    data_stream = FilterSources(data_stream, sources=('embeddings', 'embeddings_mask', 'pos'))
    data_stream = Mapping(data_stream, _transpose)

    # Initialization settings
    postagger.weights_init = IsotropicGaussian(0.1)
    postagger.biases_init = Constant(0.0)
    postagger.push_initialization_config()

    # Build the cost computation graph
    embeddings = tensor.tensor3("embeddings")
    embeddings_mask = tensor.matrix("embeddings_mask")
    pos = tensor.lmatrix("pos")
    batch_cost = postagger.cost(embeddings, embeddings_mask, pos).sum()
    batch_size = embeddings.shape[1].copy(name="batch_size")
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
    max_length = embeddings.shape[0].copy(name="max_length")
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
    data_stream = Padding(data_stream, mask_sources=('embeddings',))
    data_stream = FilterSources(data_stream, sources=('words', 'embeddings', 'embeddings_mask'))
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

    for i_words, i_embeddings, i_embeddings_mask in data_stream.get_epoch_iterator():
        o_pos, o_mask = tag_fn(i_embeddings, i_embeddings_mask)
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
    data_stream = Padding(data_stream, mask_sources=('embeddings', 'pos'))
    data_stream = FilterSources(data_stream, sources=('embeddings', 'embeddings_mask', 'pos'))
    data_stream = Mapping(data_stream, _transpose)

    embeddings = tensor.tensor3('embeddings')
    embeddings_mask = tensor.matrix('embeddings_mask')
    pos, out_mask = postagger.apply(embeddings, embeddings_mask)

    model = Model(pos)
    with open(save_path, 'rb') as f:
        model.set_parameter_values(load_parameters(f))

    tag_fn = theano.function(inputs=[embeddings, embeddings_mask], outputs=[pos, out_mask])

    npos = len(pos_voc)

    # total = 0
    # matches = 0

    gold_table = numpy.zeros((npos,))
    pred_table = numpy.zeros((npos,))
    hit_table = numpy.zeros((npos,))

    for i_embeddings, i_embeddings_mask, i_pos_idx in data_stream.get_epoch_iterator():
        o_pos, o_mask = tag_fn(i_embeddings, i_embeddings_mask)
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
        "embedmodel",
        help="The path to the trained word embedding model")
    parser.add_argument(
        "embedtrain",
        help="the embedding training corpus (required for both training and prediction)")
    parser.add_argument(
        "postrain",
        help="the POS training corpus (required for both training and prediction)")
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

    with open(args.embedtrain, 'r') as f:
        _, chars_voc = load_training(f)

    with open(args.postrain, 'r') as f:
        text_ds, pos, pos_voc = load_conll(f, chars_voc=chars_voc)

    embedding_dim = 100
    tagger_dim = 101

    hidden_dim = 101

    if args.recurrent_type == 'rnn':
        embedder_transition = SimpleRecurrent(activation=Tanh(), dim=embedding_dim)
    elif args.recurrent_type == 'lstm':
        embedder_transition = LSTM(dim=embedding_dim)
    elif args.recurrent_type == 'gru':
        embedder_transition = GatedRecurrent(dim=embedding_dim)
    else:
        raise ValueError('Unknown transition type: ' + args.recurrent_type)

    # TODO: Name of ContextEmbedder brick should be changed here and in embed.py.
    embedder = ContextEmbedder(len(chars_voc), embedding_dim, hidden_dim, embedder_transition, 0.0, chars_voc[' '],
                               name="tagger")
    tagger = PreembeddedPOSTagger(embedding_dim, tagger_dim, len(pos_voc), args.recurrent_type, name="tagger")

    if args.mode == "train":
        num_batches = args.num_batches
        text_enc = embed(embedder, text_ds, args.embedmodel)
        train_data = collections.OrderedDict()
        train_data['embeddings'] = text_enc
        train_data['pos'] = pos
        train_ds = IndexableDataset(train_data)
        train(tagger, train_ds, num_batches, args.taggermodel, step_rule=args.step_rule)
    elif args.mode == "predict":
        with open(args.test_file, 'r') if args.test_file is not None else sys.stdin as f:
            text_ds = load_vertical(f, chars_voc)
        test_enc = embed(embedder, text_ds, args.embedmodel)
        test_data = collections.OrderedDict()
        test_data['embeddings'] = test_enc
        test_ds = IndexableDataset(test_data)
        predict(tagger, test_ds, args.taggermodel, pos_voc)
    elif args.mode == "eval":
        with open(args.test_file, 'r') if args.test_file is not None else sys.stdin as f:
            text_ds, pos, _ = load_conll(f, chars_voc, pos_voc=pos_voc)
        test_enc = embed(embedder, text_ds, args.embedmodel)
        test_data = collections.OrderedDict()
        test_data['embeddings'] = test_enc
        test_data['pos'] = pos
        test_ds = IndexableDataset(test_data)
        evaluate(tagger, test_ds, args.taggermodel, pos_voc)


if __name__ == '__main__':
    main()