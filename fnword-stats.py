from dependency import FNWORD_POS, conll_trees

import itertools
import sys


def main():
    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s conll\n' % sys.argv[0])
        sys.exit(1)
    conll_file = sys.argv[1]

    with open(conll_file, 'r') as f:
        for nodes in conll_trees(f):
            for n in nodes:
                if n.pos in FNWORD_POS:
                    # path to next content word head
                    ptr = list(n.path_to_root())
                    path_to_content_head = list(itertools.takewhile(lambda x: x.pos in FNWORD_POS, ptr))
                    path_to_content_head.append(ptr[len(path_to_content_head)])

                    content_head = path_to_content_head[-1]

                    printable_path = ['%s-%s' % (mod.deprel, hd.pos)
                                      for mod, hd in zip([n] + path_to_content_head, path_to_content_head)]

                    # position relative to content head
                    if not content_head.is_root():
                        relative_pos = n.index - content_head.index
                    else:
                        relative_pos = None

                    print(n, relative_pos, len(path_to_content_head), printable_path, sep='\t')

if __name__ == '__main__':
    main()
