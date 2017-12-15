#! /usr/bin/env python3

'''
Project Steps:
x Acquire parallel data
x Select sentences
x Translate some into test set
X Acquire vectors
x Load vectors
x Acquire transformation matrix
x Apply transformation matrix
x Load data
x Acquire more english data --> https://www.reddit.com/r/LanguageTechnology/comments/44r0h2/is_there_a_large_and_open_english_word_corpus/
x Cluster
x Methods of meaning
Find best meaning
x Methods of clustering algorithm
Find best clustering algorithm
Test
Tune
'''
import random

from scripts import util, data
from scripts.fasttext import FastVector
from scripts.word_cluster import WordCluster

'''try gmm'''

if __name__ == '__main__':
    # v = FastVector(vector_file='./vec/wiki.ru.vec.reduced')
    # v.apply_transform('./vec/ru.txt')
    list = data.get_word_list()
    random.shuffle(list)
    for word in list:
        print('word:{}'.format(word))
        s = 0
        for i in range(5):
            print('iter {}'.format(i))
            wc = WordCluster(word=word, clusterer='kmeans', reduce_dimensions=True, normalize=True)
            wc.cluster()
            wc.print_cluster_overview()
            pred, truth = wc.test()
            f1 = util.w_f1(pred, truth)
            # print(acc)
            s += f1
        print('F1:{}'.format(s / float(5)))
