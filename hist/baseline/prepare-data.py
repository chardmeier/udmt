import os


def indir(name):
    return os.path.join('/home/nobackup/ch/historical-data', name)


def outdir(name):
    return os.path.join('/home/nobackup/ch/historical-data/baseline/corpus', name)


def process_parallel(infilename, srcfilename, tgtfilename):
    with open(infilename, 'r') as infile, \
            open(srcfilename, 'w') as srcfile, \
            open(tgtfilename, 'w') as tgtfile:
        for line in infile:
            src, tgt = line.rstrip('\n').split('\t')
            if not src or not tgt:
                continue
            print(' '.join(src), file=srcfile)
            print(' '.join(tgt), file=tgtfile)


def process_monolingual(infilename, outfilename):
    with open(infilename, 'r') as infile, open(outfilename, 'w') as outfile:
        for line in infile:
            print(' '.join(line.rstrip('\n')), file=outfile)


train_sv = indir('swedish.hs-sv.train.hssv')
train_sv_hist = outdir('swedish.hs-sv.train.histsv')
train_sv_norm = outdir('swedish.hs-sv.train.normsv')

dev_sv = indir('swedish.hs-sv.dev.hssv')
dev_sv_hist = outdir('swedish.hs-sv.dev.histsv')
dev_sv_norm = outdir('swedish.hs-sv.dev.normsv')

test_sv = indir('swedish.hs-sv.test.hssv')
test_sv_hist = outdir('swedish.hs-sv.test.histsv')
test_sv_norm = outdir('swedish.hs-sv.test.normsv')

sv_langmodel = indir('swedish.langmodel')
sv_langmodel_out = outdir('swedish.langmodel.normsv')

process_parallel(train_sv, train_sv_hist, train_sv_norm)
process_parallel(dev_sv, dev_sv_hist, dev_sv_norm)
process_parallel(test_sv, test_sv_hist, test_sv_norm)
process_monolingual(sv_langmodel, sv_langmodel_out)

train_de = indir('german.de-hs.train.hsde')
train_de_hist = outdir('german.de-hs.train.histde')
train_de_norm = outdir('german.de-hs.train.normde')

dev_de = indir('german.de-hs.dev.hsde')
dev_de_hist = outdir('german.de-hs.dev.histde')
dev_de_norm = outdir('german.de-hs.dev.normde')

test_de = indir('german.de-hs.test.hsde')
test_de_hist = outdir('german.de-hs.test.histde')
test_de_norm = outdir('german.de-hs.test.normde')

de_langmodel = indir('german.langmodel')
de_langmodel_out = outdir('german.langmodel.normde')

process_parallel(train_de, train_de_hist, train_de_norm)
process_parallel(dev_de, dev_de_hist, dev_de_norm)
process_parallel(test_de, test_de_hist, test_de_norm)
process_monolingual(de_langmodel, de_langmodel_out)
