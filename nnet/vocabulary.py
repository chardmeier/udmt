import h5py
import heapq
import itertools


class Vocabulary(object):
    def __init__(self, with_none=True):
        if with_none:
            self.dict = {None: 0}
            self.reverse = [None]
            self.counts = [0]
        else:
            self.dict = {}
            self.reverse = []
            self.counts = []
        self.counting = True

    def __len__(self):
        return len(self.dict)

    def from_list(self, voclist, counts=None):
        self.reverse = voclist[:]
        if counts:
            self.counts = counts
        else:
            self.counts = [0] * len(voclist)
        self.dict = {w: i for i, w in enumerate(self.reverse)}

    def lookup(self, word, append=False):
        if word in self.dict:
            idx = self.dict[word]
            if self.counting:
                self.counts[idx] += 1
            return idx
        else:
            if append:
                vocid = len(self.dict)
                self.reverse.append(word)
                self.counts.append(0)
                self.dict[word] = vocid
            else:
                if self.reverse[0] is None:
                    vocid = 0
                else:
                    raise KeyError('Out of vocabulary word: %s' % word)
            if self.counting:
                self.counts[vocid] += 1
            return vocid

    def limit_vocabulary(self, cutoff):
        has_none = self.reverse[0] is None
        retain = [t[1] for t in heapq.nlargest(cutoff,
                                               zip(self.counts[int(has_none):], itertools.count(start=int(has_none))))]
        newvoc = [None] + [self.reverse[i] for i in retain]
        vocmap = {old: i + 1 for i, old in enumerate(retain)}
        vocmap[0] = 0
        self.from_list(newvoc)
        return vocmap


class GenderDatabase(object):
    def __init__(self):
        self.values = None
        self.gender = {}
        self.group_name = None

    def from_hdf5(self, group):
        ds = group[self.group_name]
        self.gender = {ds[i, 0]: ds[i, 1] for i in range(ds.shape[0])}

    def save_hdf5(self, group):
        strtype = h5py.special_dtype(vlen=str)
        if self.group_name in group.keys():
            del group[self.group_name]
        ds = group.create_dataset(self.group_name, (len(self.gender), 2), dtype=strtype)
        for i, keyval in enumerate(self.gender.items()):
            ds[i, 0], ds[i, 1] = keyval

    def lookup(self, lemma):
        return self.gender.get(lemma, None)


class LefffGender(GenderDatabase):
    def __init__(self):
        super(LefffGender, self).__init__()
        self.values = ['m', 'f']
        self.group_name = 'lefff_gender'

    def from_file(self, file):
        with open(file, 'r') as f:
            for line in f:
                fields = line.split('\t')
                pos = fields[1]
                if not pos.startswith('n'):
                    continue
                gennum = fields[3]
                if not gennum or gennum[0] not in {'m', 'f'}:
                    continue
                lemma = fields[2]
                gender = gennum[0]
                if lemma in self.gender and self.gender[lemma] != gender:
                    self.gender[lemma] = None
                else:
                    self.gender[lemma] = gender


class SMORGender(GenderDatabase):
    def __init__(self):
        super(SMORGender, self).__init__()
        self.values = ['m', 'f', 'n']
        self.group_name = 'smor_gender'

    def from_file(self, file):
        with open(file, 'r') as f:
            for line in f:
                fields = line.split('\t')
                self.gender[fields[0]] = fields[1]
