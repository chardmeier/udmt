FNWORD_POS = {'ADP', 'AUX', 'CONJ', 'DET', 'PART', 'PRON', 'PUNCT', 'SCONJ'}


def conll_trees(file):
    while True:
        tree = load_conll(file)
        if tree is None:
            break
        yield tree


def load_conll(f):
    nodes = []
    root = Node(nodes, None)
    nodes.append(root)
    while True:
        line = f.readline().rstrip('\n')
        if not line:
            break
        nodes.append(Node(nodes, line))
    for n in nodes[1:]:
        n.head = nodes[n.head_idx]
        n.head.children.append(n)
    return nodes


class Node:
    def __init__(self, nodes, conll_line):
        self.nodes = nodes

        # set in load_conll
        self.children = []
        self.head = None

        if conll_line is not None:
            fields = conll_line.split('\t')
            self.index = int(fields[0])
            self.token = fields[1]
            self.pos = fields[5]
            self.head_idx = int(fields[9])
            self.deprel = fields[11]
        else:
            self.index = 0
            self.token = '[ROOT]'
            self.pos = 'ROOT'
            self.head_idx = None
            self.deprel = ''

    def is_root(self):
        return self.head is None

    def path_to_root(self):
        n = self
        while n.head is not None:
            n = n.head
            yield n

    def __str__(self):
        return '[%d:%s/%s-%s]' % (self.index, self.token, self.pos, self.deprel)
