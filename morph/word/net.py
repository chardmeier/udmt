from dependency import conll_trees
from nnet.nnet import create_optimizer, Net
from nnet.vocabulary import Vocabulary

from keras.layers import Activation, Dense, Dropout, Embedding, GRU, LSTM, RepeatVector, TimeDistributed
from keras.models import Sequential

import h5py
import json
import logging
import numpy
import sys
import theano


class Configuration:
    def __init__(self, json_str=None):
        self.prop = {'max_sequence': 40,

                     'encoder/embedding': 10,
                     'encoder/size': 100,
                     'encoder/dropout': 0.2,
                     'encoder/type': 'lstm',

                     'decoder/size': 100,
                     'decoder/dropout': 0.2,
                     'decoder/type': 'lstm',

                     'output_file': 'final.hdf5',

                     'batch_size': 16,
                     'nepochs': 10,
                     'optimizer': 'rmsprop',
                     'learning_rate': .001}

        if json_str is not None:
            config = json.loads(json_str)
            self.prop.update(config)

    def get(self, x):
        return self.prop[x]

    def to_json(self):
        return json.dumps(self.prop, indent='\t')


class Lemmatiser:
    def __init__(self, voc, config):
        self.voc = voc
        self.config = config
        self.model = self._setup_model()

    def _setup_model(self):
        vocsize = len(self.voc)

        types = {'lstm': LSTM, 'gru': GRU}

        enc_embedding = self.config.get('encoder/embedding')
        enc_size = self.config.get('encoder/size')
        enc_dropout = self.config.get('encoder/dropout')
        enc_type = types[self.config.get('encoder/type')]

        dec_type = types[self.config.get('decoder/type')]
        dec_size = self.config.get('decoder/size')
        dec_dropout = self.config.get('decoder/dropout')

        max_outlen = self.config.get('max_sequence')

        opt = create_optimizer(self.config)

        model = Sequential()
        model.add(Embedding(vocsize, enc_embedding, mask_zero=True))
        model.add(enc_type(enc_size, return_sequences=False))
        model.add(Dropout(enc_dropout))
        model.add(RepeatVector(max_outlen))
        model.add(dec_type(dec_size, return_sequences=True))
        model.add(Dropout(dec_dropout))
        model.add(TimeDistributed(Dense(vocsize)))
        model.add(Activation('softmax'))

        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy')

        return model

    def save_metadata(self, file):
        with h5py.File(file, 'a') as f:
            grp = f.require_group('/udmt/word_lemmatiser')
            grp.attrs['config'] = self.config.to_json()
            vocgrp = grp.require_group('vocabularies')
            strtype = h5py.special_dtype(vlen=str)
            for name in vocgrp.keys():
                del vocgrp[name]
            ds = vocgrp.create_dataset('chars', (len(self.voc),), dtype=strtype)
            ds[:] = self.voc.reverse[:]
            f.flush()


class LemmatiserDataset:
    def __init__(self, config, voc=None):
        self.config = config

        if voc is None:
            self.voc = Vocabulary()
            self.append_voc = True
        else:
            self.voc = voc
            self.append_voc = False

        self._token_boundary = self.voc.lookup(' ', self.append_voc)

        self.max_seqlen = config.get('max_sequence')
        self.seqlen = 0

        self.dtype = [('token', numpy.int32, self.max_seqlen), ('lemma', numpy.int32, (self.max_seqlen, 1))]
        self.data = numpy.zeros((0,), dtype=self.dtype)

    def __len__(self):
        return len(self.data)

    def load_data(self, file):
        data = []
        for tree in conll_trees(file):
            for n in tree[1:]:
                if len(n.token) > self.max_seqlen - 1 or len(n.lemma) > self.max_seqlen - 1:
                    continue
                tok_chars = list(self.voc.lookup(c, self.append_voc) for c in n.token)
                tok_chars.append(self._token_boundary)
                lemma_chars = list(self.voc.lookup(c, self.append_voc) for c in n.lemma)
                lemma_chars.append(self._token_boundary)
                data.append((tok_chars, lemma_chars))

        self.data = numpy.zeros((len(data),), dtype=self.dtype)
        for i, (tok_chars, lemma_chars) in enumerate(data):
            self.data['token'][i, 0:len(tok_chars)] = tok_chars[:]
            self.data['lemma'][i, 0:len(lemma_chars), 0] = lemma_chars[:]

    def make_batch(self, perm):
        batch = LemmatiserDataset(self.config, voc=self.voc)
        batch.data = self.data[perm]
        return batch

    def get_inputs(self):
        return self.data['token']

    def get_labels(self):
        return self.data['lemma']


def seq_to_str(seq, voc):
    rev = voc.reverse[:]
    rev[0] = '_'
    outstr = ''
    for c in numpy.nditer(seq):
        if c == 1:
            break
        outstr += rev[c]
    return outstr


def output_predictions(dataset, inpred):
    inp = dataset.get_inputs()
    truth = dataset.get_labels()
    pred = numpy.argmax(inpred, axis=-1)
    for i in range(pred.shape[0]):
        inp_str = seq_to_str(inp[i, :], dataset.voc)
        truth_str = seq_to_str(truth[i, :, 0], dataset.voc)
        pred_str = seq_to_str(pred[i, :], dataset.voc)
        print('%s\t%s\t%s' % (inp_str, truth_str, pred_str))


def main():
    theano.config.optimizer = 'None'
    theano.config.exception_verbosity = 'high'
    if len(sys.argv) != 5:
        sys.stderr.write('Usage: %s config.json train.conllu val.conllu test.conllu\n' % sys.argv[0])
        sys.exit(1)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(''message)s')

    config_file = sys.argv[1]
    train_file = sys.argv[2]
    val_file = sys.argv[3]
    test_file = sys.argv[4]

    with open(config_file, 'r') as f:
        config = Configuration(f.read())

    logging.info('Loading training data from %s' % train_file)
    train = LemmatiserDataset(config)
    with open(train_file, 'r') as f:
        train.load_data(f)

    logging.info('Loading validation data from %s' % val_file)
    val = LemmatiserDataset(config, voc=train.voc)
    with open(val_file, 'r') as f:
        val.load_data(f)

    logging.info('Loading test data from %s' % val_file)
    test = LemmatiserDataset(config, voc=train.voc)
    with open(test_file, 'r') as f:
        test.load_data(f)

    logging.info('Creating model')
    net = Net(Lemmatiser(train.voc, config), config)

    logging.info('Training model')
    net.train(train, val)

    logging.info('Saving model')
    net.save(config_file + '.hdf5')

    logging.info('Making predictions for test file.')
    test_pred = net.predict(test)
    output_predictions(test, test_pred)

    logging.info('Done.')


if __name__ == "__main__":
    main()

