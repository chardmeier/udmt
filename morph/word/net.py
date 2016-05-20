from dependency import conll_trees
from nnet.nnet import create_optimizer, Net
from nnet.vocabulary import Vocabulary

from keras.layers import Dropout, Embedding, GRU, LSTM, RepeatVector
from keras.models import Sequential

import h5py
import json
import logging
import numpy
import sys


class Configuration:
    def __init__(self, json_str=None):
        self.prop = {'max_sequence': 40,
                     'encoder/embedding': 10,
                     'encoder/size': 40,
                     'encoder/dropout': 0.2,
                     'encoder/type': 'lstm',
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

        max_outlen = self.config.get('max_sequence')

        opt = create_optimizer(self.config.get('optimizer'))

        model = Sequential()
        model.add(Embedding(vocsize, enc_embedding))
        model.add(enc_type(enc_size, return_sequences=False))
        model.add(Dropout(enc_dropout))
        model.add(RepeatVector(max_outlen))
        model.add(dec_type(vocsize, return_sequences=True))

        model.compile(optimizer=opt, loss='categorical_crossentropy')

        return model

    def save_metadata(self, file):
        with h5py.File(file, 'a') as f:
            grp = f.require_group('/udmt/word_lemmatiser')
            grp.attrs['config'] = self.config.to_json()
            vocgrp = grp.require_group('vocabularies')
            strtype = h5py.special_dtype(vlen=str)
            for name in vocgrp.keys():
                del vocgrp[name]
            for name, voc in self.voc.items():
                ds = vocgrp.create_dataset(name, (len(voc),), dtype=strtype)
                ds[:] = voc.reverse[:]
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

        self._token_boundary = self.voc['chars'].lookup(' ', self.append_voc)

        self.max_seqlen = config.get('max_sequence')
        self.seqlen = 0

        self.dtype = [('token', numpy.int32, self.max_seqlen), ('lemma', numpy.int32, self.max_seqlen)]
        self.data = numpy.zeros((0,), dtype=self.dtype)

    def __len__(self):
        return len(self.data)

    def load_data(self, file):
        data = []
        for tree in conll_trees(file):
            for n in tree[1:]:
                if len(n.token) > self.max_seqlen or len(n.lemma) > self.max_seqlen:
                    continue
                tok_chars = list(self.voc.lookup(c, self.append_voc) for c in n.token)
                tok_chars.append(self._token_boundary)
                lemma_chars = list(self.voc.lookup(c, self.append_voc) for c in n.lemma)
                lemma_chars.append(self._token_boundary)
                data.append((tok_chars, lemma_chars))

        self.data = numpy.zeros((len(data),), dtype=self.dtype)
        for i, (tok_chars, lemma_chars) in enumerate(data):
            self.data['token'][i, 0:len(tok_chars)] = tok_chars[:]
            self.data['lemma'][i, 0:len(lemma_chars)] = lemma_chars[:]

    def make_batch(self, perm):
        batch = LemmatiserDataset(self.config, voc=self.voc)
        batch.data = self.data[perm]
        return batch

    def get_inputs(self):
        return self.data['token']

    def get_labels(self):
        return self.data['lemma']


def main():
    if len(sys.argv) != 4:
        sys.stderr.write('Usage: %s config.json train.conllu val.conlllu\n' % sys.argv[0])
        sys.exit(1)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(''message)s')

    config_file = sys.argv[1]
    train_file = sys.argv[2]
    val_file = sys.argv[3]

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

    logging.info('Creating model')
    net = Net(LemmatiserDataset(train.voc, config), config)

    logging.info('Training model')
    net.train(train, val)

    logging.info('Saving model')
    net.save(config.get('output_file'))

    logging.info('Done.')


if __name__ == "__main__":
    main()

