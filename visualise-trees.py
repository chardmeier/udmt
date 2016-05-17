from dependency import conll_trees, tree_to_udmt, UDMTException

import re
import sys


tex_header = r'''
\documentclass[a4paper,12pt]{scrartcl}
\usepackage[utf8]{inputenc}
\usepackage{fourier}
\usepackage{tikz}
\usepackage{tikz-dependency}
\begin{document}'''

tex_footer = r'\end{document}'


def tex_escape(s):
    return re.sub(r'[^A-Za-z0-9_.,:;/-]', '=', s)


def show_tree(udmt_tree, f):
    print(r'\begin{dependency}', file=f)
    print(r'\begin{deptext}', file=f)
    print(r'\&'.join(str(i) for i in range(len(udmt_tree))) + r'\\', file=f)
    print(r'\&'.join(tex_escape(n.pos) for n in udmt_tree) + r'\\', file=f)
    print(r'\&'.join(tex_escape(n.token) for n in udmt_tree) + r'\\', file=f)
    print(r'\end{deptext}', file=f)
    for head in udmt_tree:
        for mod in head.children:
            print(r'\depedge{%d}{%d}{%s}' % (mod.index + 1, head.index + 1, tex_escape(mod.deprel)), file=f)
    print(r'\end{dependency}', file=f)
    udmt_features = [(n.index, n.udmt_features) for n in udmt_tree if n.udmt_features]
    if udmt_features:
        print(r'\begin{tabular}{cl}', file=f)
        for feat in udmt_features:
            print(r'%d&%s\\' % feat, file=f)
        print(r'\end{tabular}', file=f)


def main():
    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s conll' % sys.argv[0])
        sys.exit(1)

    conll_file = sys.argv[1]

    outfile = sys.stdout
    outfile.write(tex_header)

    with open(conll_file, 'r') as f:
        for sentno, tree in enumerate(conll_trees(f)):
            try:
                udmt_tree = tree_to_udmt(tree)
            except UDMTException as e:
                print("Sentence %d: %s" % (sentno, e), file=sys.stderr)
            else:
                show_tree(udmt_tree, outfile)

    outfile.write(tex_footer)

if __name__ == "__main__":
    main()
