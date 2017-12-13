import numpy as np
from scripts.data import get_english_domain
from scripts.fasttext import FastVector

VECTOR_LENGTH = 300


def get_meanings():
    return {'average': average}


def average(data, word):
    model = FastVector('./vec/en.reduced.v2')
    final_matrix = np.zeros([len(data), VECTOR_LENGTH])
    misses = 0
    for i, line in enumerate(data):
            line = [l for l in line if l != word]
            sent_matrix = np.zeros([len(line), VECTOR_LENGTH])
            for j, word in enumerate(line):
                try:
                    sent_matrix[j] = model[word]
                except KeyError as e:
                    misses += 1
            ave = sent_matrix.mean(axis=0)  # check this dimensionality here
            final_matrix[i] = ave
    print('Vector misses: {0} (Rate: {1:0.3f} words / sample)'.format(misses, (misses / len(data))))
    return final_matrix
