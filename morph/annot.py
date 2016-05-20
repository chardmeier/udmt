from morph.net import AnalyserDataset, Configuration, load_analyser

import logging
import sys


def main():
    if len(sys.argv) != 3:
        sys.stderr.write('Usage: %s model.hdf5 test.conllu\n' % sys.argv[0])
        sys.exit(1)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(''message)s')

    model_file = sys.argv[1]
    test_file = sys.argv[2]

    net = load_analyser(model_file)

    logging.info('Loading test data from %s' % test_file)
    test = AnalyserDataset(net.config, voc=net.voc)
    with open(test_file, 'r') as f:
        test.load_data(f)

    predictions = net.predict(test)

    logging.info('Done.')


if __name__ == "__main__":
    main()