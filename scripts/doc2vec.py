import glob
import random

from gensim.models import doc2vec
from collections import namedtuple

from gensim.utils import tokenize


def make_tokens(file):
    return [tokenize(f.strip('\n'), lower=True, deacc=True) for f in open(file).readlines()]

    # with open('./dat/all/gen{}.tokenize'.format(i), 'w+') as f:
    #     for l in lines:
    #         f.write(l + '\n')


def make_model(epochs=10):
    model = doc2vec.Doc2Vec(size=100,  # Model initialization
                            window=8,
                            min_count=10,
                            workers=4)

    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')

    alpha_val = 0.025  # Initial learning rate
    min_alpha_val = 1e-4  # Minimum for linear learning rate decay
    alpha_delta = (alpha_val - min_alpha_val) / (epochs - 1)

    # This is gon' be big
    F = 0

    # Deterministic for labels
    file_list = glob.glob('./dat/billion_corpus/all/gen*')
    print(file_list)

    for e in range(epochs):

        model.alpha, model.min_alpha = alpha_val, alpha_val
        I = 0
        for file in file_list:
            docs = []
            lines = make_tokens(file)
            for l in lines:
                tags = [I]
                docs.append(analyzedDocument(l, tags))
                I += 1

            model.train(docs)
            print('Files Trained: {}'.format(F))
            F += 1
            print('Saving')
            model.save('model.gensim')
        alpha_val -= alpha_delta
        print('Epochs: {}'.format(e))


if __name__ == '__main__':
    make_model()
