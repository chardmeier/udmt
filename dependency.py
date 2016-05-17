import copy
import itertools


FNWORD_POS = {'ADP', 'AUX', 'CONJ', 'DET', 'PART', 'PRON', 'PUNCT', 'SCONJ'}


def conll_trees(conll_file):
    for key, group in itertools.groupby(conll_file, lambda l: l == '\n'):
        if not key:
            nodes = []
            nodes.append(Node(nodes))
            for line in group:
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


def tree_to_udmt_traverse(root):
    udmt_nodes = []

    udmt = copy.copy(root)
    udmt.nodes = udmt_nodes
    udmt_nodes.append(udmt)

    for n in root.children:
        if n.is_function_word():
            fw = tree_to_udmt_traverse_fnword(n)
            udmt.udmt_features.append(fw)
        else:
            udmt_nodes.extend(tree_to_udmt_traverse(n))

    return udmt_nodes


def tree_to_udmt(nodes):
    if not nodes:
        return nodes

    udmt_nodes = tree_to_udmt_traverse(nodes[0])
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
            self.pos = fields[5]
            self.head_idx = int(fields[9])
            self.deprel = fields[11]
        else:
            self.index = 0
            self.word_idx = 0
            self.token = '[ROOT]'
            self.pos = 'ROOT'
            self.head_idx = None
            self.deprel = ''

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
