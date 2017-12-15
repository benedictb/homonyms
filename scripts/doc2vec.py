import glob
import pickle

from gensim.models import doc2vec
from collections import namedtuple, Counter

from gensim.utils import tokenize
import tqdm


def make_tokens(file_list):
    print('Tokenizing...')
    tokens = []
    for file in file_list:
        tokens+=[list(tokenize(f.strip('\n'), lower=True, deacc=True)) for f in open(file).readlines()]
    print('DONE')
    print(tokens)
    return tokens


def build_vocab_er(file_list):
    try:
        with open('vocab.p', 'rb') as handle:
            c = pickle.load(handle)
    except OSError:
        c = Counter()
        for file in tqdm.tqdm(file_list, total=len(file_list)):
            for l in open(file):
                c.update(l.strip('\n').split(' '))
        with open('vocab.p', 'wb') as handle:
            pickle.dump(c, handle, protocol=pickle.HIGHEST_PROTOCOL)

    del c['']
    return c


def make_model(epochs=10):
    model = doc2vec.Doc2Vec(size=100,  # Model initialization
                            window=8,
                            min_count=8,
                            workers=4)
    # docvecs_mapfile='./mapfile.map')

    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')


    # Deterministic for labels
    # file_list = glob.glob('./dat/billion_corpus/all/*')
    file_list = ['./test.txt']
    print(file_list)

    word_count = build_vocab_er(file_list)
    model.build_vocab_from_freq(word_count)

    # yay ram
    tokens = make_tokens(file_list)
    docs = list([(analyzedDocument(l, [i]) for l, i in zip(tokens, range(len(tokens))))])

    for e in range(epochs):
        model.train(docs, total_words=sum(word_count.values()), epochs=1)
        print('Epochs Trained: {}'.format(e))
        print('Saving')
        model.save('model.gensim')
        print('Saved')


        # docs = []
        # lines = make_tokens(file)
        #
        # for l in tqdm.tqdm(lines, total=len(lines)):
        #     tags = [I]
        #     I += 1


if __name__ == '__main__':
    make_model()
