from bs4 import BeautifulSoup


def main():
    with open('/tmp/SUC_2.0_Tiger.xml', 'r') as f:
        soup = BeautifulSoup(f.read(), 'xml')
    for snt in soup.find_all('terminals'):
        for i, t in enumerate(snt.find_all('t')):
            word = t['word']
            pos = t['pos']
            morph = t['morph'].replace(' ', '|')
            lemma = t['lemma']
            misc = t['ntype'] if t['ntype'] != '--' else '_'
            conll_line = '\t'.join([
                i + 1,
                word,
                lemma,
                pos,
                pos,
                morph,
                '_',
                '_',
                '_',
                misc
            ])
            print(conll_line)
        print()

if __name__ == '__main__':
    main()
