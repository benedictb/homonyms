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
Methods of meaning
Find best meaning
Methods of clustering algorithm
Find best clustering algorithm
Test
Tune
'''
import random

from scripts import util, data
from scripts.fasttext import FastVector
from scripts.word_cluster import WordCluster

'''
Issues that need resolving
- Deal with multi sentences, in training and in testing.
- save data binaries for each word, processing time is probably increasing
- normalize vectors OR just do cosine similarity as cluster mechanism 
- if we do reduce dimensionality, we'd have to save the model and then apply the same reduction to new samples
    before finding the best cluster fit
'''

if __name__ == '__main__':
    # v = FastVector(vector_file='./vec/wiki.ru.vec.reduced')
    # v.apply_transform('./vec/ru.txt')
    # word = random.choice(data.get_word_list())
    # wc.print_clusters()

    wc = WordCluster(word='organ')
    wc.cluster()
    wc.explore()