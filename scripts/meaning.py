import fasttext

# tbd
import numpy as np

VECTOR_LENGTH = 300


def average(data, word):
    model = fasttext.load_model('wiki.in.vec')
    final_matrix = np.zeros([len(data), VECTOR_LENGTH])
    for i, line in enumerate(data):
        line = [l for l in line if l != word]
        sent_matrix = np.zeros([len(line), VECTOR_LENGTH])
        for j, word in enumerate(line):
            sent_matrix[j] = model[word]
        final_matrix[i] = sent_matrix.mean(axis=1)  # check this dimensionality here
    return final_matrix
