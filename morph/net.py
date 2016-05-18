from keras import backend as K

from keras.models import Model, Sequential
from keras.layers import Activation, Dense, Input, Layer, LSTM, TimeDistributed

from dependency import conll_trees
from nnet.nnet import create_optimizer, Net
from nnet.vocabulary import Vocabulary

import h5py
import json
import logging
import numpy
import sys


class CharToWord(Layer):
    def __init__(self, token_boundary, **kwargs):
        self.token_boundary = token_boundary
        super(CharToWord, self).__init__(**kwargs)

    def compute_mask(self, inputs, input_mask=None):
        chars = inputs[1]
        return K.not_equal(chars[:, :, self.token_boundary], 0.0)

    def get_output_shape_for(self, input_shape):
        return input_shape[0]

    def call(self, x, mask=None):
        return x[0]


class Analyser:
    def __init__(self, voc, config):
        self.voc = voc
        self.config = config
        self.model = self._setup_model()

    def _setup_model(self):
        i_chars = Input(shape=(None, len(self.voc['chars'])), name='i_chars')

        token_boundary = self.voc['chars'].lookup(' ')

        wordlayersize = self.config.get('words/size')

        x = LSTM(self.config.get('chars/size'), return_sequences=True, consume_less='gpu')(i_chars)
        x = CharToWord(token_boundary=token_boundary)([x, i_chars])
        x = LSTM(wordlayersize, return_sequences=True, consume_less='gpu')(x)

        net_features = Sequential()
        net_features.add(Dense(len(self.voc['features']), input_shape=(wordlayersize,)))
        net_features.add(Activation('sigmoid'))
        o_features = TimeDistributed(net_features, name='features')(x)

        net_pos = Sequential()
        net_pos.add(Dense(len(self.voc['pos']), input_shape=(wordlayersize,)))
        net_pos.add(Activation('softmax'))
        o_pos = TimeDistributed(net_pos, name='pos')(x)

        model = Model(input=i_chars, output=[o_features, o_pos])

        opt = create_optimizer(self.config)
        model.compile(optimizer=opt,
                      loss={'features': 'binary_crossentropy', 'pos': 'categorical_crossentropy'},
                      metrics=['accuracy'])
        return model

    def save_metadata(self, file):
        with h5py.File(file, 'a') as f:
            grp = f.require_group('/udmt/morph_analyser')
            grp.attrs['config'] = self.config.to_json()
            vocgrp = grp.require_group('vocabularies')
            strtype = h5py.special_dtype(vlen=str)
            for name in vocgrp.keys():
                del vocgrp[name]
            for name, voc in self.voc.items():
                ds = vocgrp.create_dataset(name, (len(voc),), dtype=strtype)
                ds[:] = voc.reverse[:]
            f.flush()


class AnalyserDataset:
    def __init__(self, config, voc=None):
        self.config = config

        if voc is None:
            self.voc = {'chars': Vocabulary(),
                        'features': Vocabulary(),
                        'pos': Vocabulary()}
            self.append_voc = True
        else:
            self.voc = voc
            self.append_voc = False

        self._token_boundary = self.voc['chars'].lookup(' ', self.append_voc)

        self.max_seqlen = config.get('max_sequence')
        self.seqlen = 0
        self.data = []

    def __len__(self):
        return len(self.data)

    def load_data(self, file):
        self.data = []
        for tree in conll_trees(file):
            chars = []
            for n in tree[1:(self.max_seqlen + 1)]:
                idx_pos = self.voc['pos'].lookup(n.pos, self.append_voc)
                idx_features = [self.voc['features'].lookup(ft, self.append_voc) for ft in n.features]
                chars.extend((self.voc['chars'].lookup(c, self.append_voc),) for c in n.token)
                chars.append((self._token_boundary, idx_pos, idx_features))
            self.data.append(chars)
        self.seqlen = max(len(x) for x in self.data)

    def make_batch(self, perm):
        batch = AnalyserDataset(self.config, voc=self.voc)
        batch.data = [self.data[i] for i in numpy.nditer(perm)]
        batch.seqlen = max(len(x) for x in batch.data)
        return batch

    def get_inputs(self):
        charvocsize = len(self.voc['chars'])
        chars = numpy.zeros((len(self.data), self.seqlen, charvocsize))
        for i, snt in enumerate(self.data):
            for j, c in enumerate(snt):
                chars[i, j, c[0]] = 1.0
        return chars

    def get_labels(self):
        posvocsize = len(self.voc['pos'])
        pos = numpy.zeros((len(self.data), self.seqlen, posvocsize))
        featvocsize = len(self.voc['features'])
        feats = numpy.zeros((len(self.data), self.seqlen, featvocsize))
        for i, snt in enumerate(self.data):
            for j, c in enumerate(snt):
                if len(c) < 3:
                    continue
                pos[i, j, c[1]] = 1.0
                for ft in c[2]:
                    feats[i, j, ft] = 1.0
        return {'pos': pos, 'features': feats}


class Configuration:
    def __init__(self, file=None):
        self.prop = {'max_sequence': 200,
                     'chars/size': 100,
                     'words/size': 100,

                     'output_file': 'final.hdf5',

                     'batch_size': 16,
                     'nepochs': 10,
                     'optimizer': 'rmsprop',
                     'learning_rate': .001}

        with open(file, 'r') as f:
            config = json.load(f)

        self.prop.update(config)

    def get(self, x):
        return self.prop[x]

    def to_json(self):
        return json.dumps(self.prop, indent='\t')


def main():
    if len(sys.argv) != 4:
        sys.stderr.write('Usage: %s config.json train.conllu val.conlllu\n' % sys.argv[0])
        sys.exit(1)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(''message)s')

    config_file = sys.argv[1]
    train_file = sys.argv[2]
    val_file = sys.argv[3]

    config = Configuration(config_file)

    logging.info('Loading training data from %s' % train_file)
    train = AnalyserDataset(config)
    with open(train_file, 'r') as f:
        train.load_data(f)

    logging.info('Loading validation data from %s' % val_file)
    val = AnalyserDataset(config, voc=train.voc)
    with open(val_file, 'r') as f:
        val.load_data(f)

    logging.info('Creating model')
    net = Net(Analyser(train.voc, config), config)

    logging.info('Training model')
    net.train(train, val)

    logging.info('Saving model')
    net.save(config.get('output_file'))

    logging.info('Done.')


if __name__ == "__main__":
    main()
