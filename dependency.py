import copy
import itertools
import re


FNWORD_POS = {'ADP', 'AUX', 'CONJ', 'DET', 'PART', 'PRON', 'PUNCT', 'SCONJ'}


def conll_trees(conll_file):
    for key, group in itertools.groupby(conll_file, lambda l: l == '\n'):
        if not key:
            nodes = []
            nodes.append(Node(nodes))
            for line in group:
                # Skip comments and nodes that span multiple tokens
                if re.match(r'#|[0-9]+-[0-9]+', line):
                    continue
                nodes.append(Node(nodes, conll_line=line.rstrip('\n')))
            for n in nodes[1:]:
                n.head = nodes[n.head_idx]
                n.head.children.append(n)
            yield nodes


class UDMTException(Exception):
    pass


def tree_to_udmt_traverse_fnword(n):
    myself = [(n.token, n.pos, n.deprel)]
    if not n.children:
        return myself
    elif len(n.children) == 1:
        my_child = n.children[0]
        if not my_child.is_function_word():
            raise UDMTException("Function word with content word modifier: %s <- %s" % (n, my_child))
        children = tree_to_udmt_traverse_fnword(my_child)
        return myself + children
    else:
        raise UDMTException("Function word with multiple children: %s <- %s" % (n, [str(n) for n in n.children]))


def tree_to_udmt_traverse(root, udmt_nodes):
    udmt = copy.copy(root)
    udmt.children = []
    udmt.udmt_features = []

    udmt.nodes = udmt_nodes
    udmt_nodes.append(udmt)

    for n in root.children:
        if n.is_function_word():
            fw = tree_to_udmt_traverse_fnword(n)
            udmt.udmt_features.append(fw)
        else:
            child = tree_to_udmt_traverse(n, udmt_nodes)
            child.head = udmt
            udmt.children.append(child)

    return udmt


def tree_to_udmt(nodes):
    if not nodes:
        return nodes

    udmt_nodes = []
    tree_to_udmt_traverse(nodes[0], udmt_nodes)
    udmt_nodes.sort(key=lambda x: x.word_idx)

    for i, n in enumerate(udmt_nodes):
        n.index = i
        for c in n.children:
            c.head_idx = i

    return udmt_nodes


class Node:
    def __init__(self, nodes, conll_line=None):
        self.nodes = nodes

        self.udmt_features = []

        # set in load_conll
        self.children = []
        self.head = None

        if conll_line is not None:
            fields = conll_line.split('\t')
            self.index = int(fields[0])
            self.word_idx = self.index
            self.token = fields[1]
            self.lemma = fields[2]
            self.pos = fields[3]
            self.xpos = fields[4]
            self.features = fields[5].split('|') if fields[5] else []
            self.head_idx = int(fields[6])
            self.deprel = fields[7]
            self.secondary_deps = fields[8]
            self.misc = fields[9]
        else:
            self.index = 0
            self.word_idx = 0
            self.token = '[ROOT]'
            self.pos = 'ROOT'
            self.xpos = 'ROOT'
            self.features = []
            self.head_idx = None
            self.deprel = ''
            self.secondary_deps = ''
            self.misc = ''

    def is_root(self):
        return self.head is None

    def is_function_word(self):
        return self.pos in FNWORD_POS or (self.pos == 'VERB' and self.deprel == 'cop')

    def path_to_root(self):
        n = self
        while n.head is not None:
            n = n.head
            yield n

    def __str__(self):
        return '[%d:%s/%s-%s:%s]' % (self.index, self.token, self.pos, self.deprel, self.head_idx)
