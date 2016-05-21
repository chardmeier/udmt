from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint
from keras.optimizers import Adadelta, Adagrad, Adam, RMSprop

import functools
import numpy
import sys


def create_optimizer(config):
    if config.prop['optimizer'] == 'rmsprop':
        opt = RMSprop(lr=config.prop['learning_rate'])
    elif config.prop['optimizer'] == 'adagrad':
        opt = Adagrad(lr=config.prop['learning_rate'])
    elif config.prop['optimizer'] == 'adadelta':
        opt = Adadelta()
    elif config.prop['optimizer'] == 'adam':
        opt = Adam()
    else:
        raise Exception('Unknown optimiser: %s' % config.prop['optimizer'])
    return opt


def batch_generator(dataset, size, repeat=True, shuffle=True):
    if shuffle:
        def indices(start, end):
            return perm[start:end]
    else:
        def indices(start, end):
            return range(start, min(end, len(dataset)))

    while True:
        if shuffle:
            perm = numpy.random.permutation(len(dataset))

        for i in range(0, len(dataset), size):
            batch = dataset.make_batch(indices(i, i + size))
            inputs = batch.get_inputs()
            labels = batch.get_labels()
            yield (inputs, labels)
        if not repeat:
            break


def to_one_hot(x, vocsize):
    y = numpy.zeros(x.shape + (vocsize,), dtype='float32')
    it = numpy.nditer(x, flags=['multi_index'])
    while not it.finished:
        y[it.multi_index + (it[0],)] = 1.0
        it.iternext()
    return y


def nn6_combine(inputs):
    antembed, v, antmap = tuple(inputs)
    # Shape of antmap is (nexmpl, nant)
    shape = K.shape(antmap)
    mapmat = K.repeat_elements(K.transpose(v), shape[0], axis=0) * antmap
    out = K.dot(mapmat, antembed)
    return out


def blackout(inputs, prob=0.5):
    if K.learning_phase() and numpy.random.random_sample() < prob:
        # The multiplication serves to keep the computation graph connected
        return K.zeros_like(inputs) * inputs
    else:
        return inputs


def compose_layers(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), reversed(functions), lambda x: x)


class Net(object):
    def __init__(self, architecture, config):
        self.config = config
        self.model = architecture.model
        self.voc = architecture.voc
        self.architecture = architecture

    def to_yaml(self):
        return self.model.to_yaml()

    def train(self, train, val, checkpoint=None):
        batch_size = self.config.prop['batch_size']
        nbatch = len(train) // batch_size + 1
        dot_interval = nbatch // 80 + 1

        callbacks = [ProgressReporter(dot_interval)]

        if checkpoint:
            callbacks.append(ModelCheckpoint(checkpoint))

        if 'balance_data' in self.config.prop:
            strategy = self.config.prop['balance_data']
            if strategy == 'strict':
                nlabels = len(train.voc['label'])
                counts = [0] * nlabels
                for i in numpy.nditer(train.data['label']):
                    counts[i] += 1
                maxcount = max(counts)
                weights = [maxcount / c for c in counts]
            else:
                raise Exception('Unknown balancing strategy: %s' % strategy)
        else:
            weights = None

        self.model.fit_generator(batch_generator(train, batch_size), len(train), self.config.prop['nepochs'],
                                 verbose=2, validation_data=batch_generator(val, batch_size),
                                 nb_val_samples=len(val), callbacks=callbacks, class_weight=weights)

    def evaluate(self, test):
        return self.model.evaluate_generator(batch_generator(test, self.config.prop['batch_size']), len(test))

    def predict(self, test):
        return self.model.predict_generator(batch_generator(test, self.config.prop['batch_size'], shuffle=False),
                                            len(test))

    def predict_labels(self, test):
        pred = self.predict(test)
        idx = pred.argmax(axis=1)
        out = [self.voc['label'].reverse[i] for i in numpy.nditer(idx)]
        return out

    def save(self, file):
        self.model.save_weights(file, overwrite=True)
        self.save_metadata(file)

    def save_metadata(self, file):
        self.architecture.save_metadata(file)


class ProgressReporter(Callback):
    def __init__(self, dotinterval, ostream=sys.stderr):
        super(ProgressReporter, self).__init__()
        self.dotinterval = dotinterval
        self.ostream = ostream
        self.counter = 0

    def on_batch_end(self, batch, logs={}):
        self.counter += 1
        if self.counter % self.dotinterval == 0:
            self.ostream.write('.')
            self.ostream.flush()

    def on_epoch_end(self, epoch, logs={}):
        self.ostream.write('\n')
        self.counter = 0
