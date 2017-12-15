import glob
import pickle

import yaml
from gensim.models import doc2vec
from collections import namedtuple, Counter

from gensim.utils import tokenize
import tqdm

config = yaml.load(open('homonyms.config'))

REBUILD = True


def make_tokens(file_list, rebuild=True):
    try:
        if rebuild:
            raise OSError()
        with open(config['tokens'], 'r') as handle:
            print('LOADING PREVIOUS')
            tokens = list([l.strip('\n').split(',') for l in handle])
    except OSError:
        print('Tokenizing...')
        tokens = []
        for file in file_list:
            print(file)
            # print(file.readlines())
            tmp = list([list(tokenize(f.strip('\n'), lower=True, deacc=True)) for f in open(file).readlines()])
            tokens += tmp
            print(tokens)
        print('DONE')
        print('SAVING')
        with open(config['tokens'], 'w+') as handle:
            for l in tokens:
                handle.write(','.join(l) + '\n')
        return list(tokens)
    return tokens


def build_vocab_er(file_list):
    try:
        with open(config['vocab'], 'rb') as handle:
            c = pickle.load(handle)
    except OSError:
        c = Counter()
        for file in tqdm.tqdm(file_list, total=len(file_list)):
            for l in open(file):
                c.update(l.strip('\n').split(' '))
        with open(config['vocab'], 'wb') as handle:
            pickle.dump(c, handle, protocol=pickle.HIGHEST_PROTOCOL)

    del c['']
    return c


def make_model(epochs=10):
    model = doc2vec.Doc2Vec(size=100,  # Model initialization
                            window=3,
                            min_count=1,
                            workers=4)
    # docvecs_mapfile='./mapfile.map')

    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')

    # Deterministic for labels
    # file_list = glob.glob('./dat/billion_corpus/all/*')
    file_list = [config['dummy']]

    word_count = build_vocab_er(file_list)
    model.build_vocab_from_freq(word_count)

    # yay ram
    tokens = make_tokens(file_list)
    docs = [(analyzedDocument(l, [i]) for l, i in zip(tokens, range(len(tokens))))]

    for e in range(epochs):
        print(type(docs))
        model.train(docs, total_words=sum(word_count.values()), epochs=1)
        print('Epochs Trained: {}'.format(e))
        print('Saving')
        model.save(config['model'])
        print('Saved')


if __name__ == '__main__':
    make_model()
