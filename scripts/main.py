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
O PROB NOT Deal with multi sentences, in training and in testing.
O I don't give a shit/ save data binaries for each word, processing time is probably increasing
- normalize vectors OR just do cosine similarity as cluster mechanism 
- metrics (besides euclidean. so cosine sim)
- make sure russian words i picked are in the vector
'''
'''
Better for sentence average
https://www.quora.com/In-Word2Vec-representation-does-the-average-vector-of-the-sentence-can-represent-the-sentence

doc2vec if you're going to do anything
https://stackoverflow.com/questions/31321209/doc2vec-how-to-get-document-vectors
'''

if __name__ == '__main__':
    # v = FastVector(vector_file='./vec/wiki.ru.vec.reduced')
    # v.apply_transform('./vec/ru.txt')
    # word = random.choice(data.get_word_list())
    # wc.print_clusters()

    wc = WordCluster(word='organ', clusterer='kmeans', reduce_dimensions=None)
    wc.cluster()
    # wc.visualize()
    wc.explore()

