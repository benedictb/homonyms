#! /usr/bin/env python3
import random

from source.word_cluster import WordCluster
from source import util, data

if __name__ == '__main__':
    # Sample use case
    word = random.choice(data.get_word_list())

    print('Word: {}'.format(word))
    wc = WordCluster(word=word, clusterer='kmeans', reduce_dimensions=True, normalize=True)
    wc.cluster()
    wc.print_cluster_overview()
    pred, truth = wc.test()
    acc = util.accuracy(pred, truth)
    print('Accuracy: {}'.format(acc))
