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

'''
Issues that need resolving
- Deal with multi sentences, in training and in testing.
- Split up util between data and util
'''

if __name__ == '__main__':
    word = random.choice(data.get_word_list())
    english = data.get_data_for_word(word)
    test = data.get_test_data_for_word(word)

    v = FastVector(vector_file='./vec/wiki.ru.vec.reduced')
    v.apply_transform('./vec/ru.txt')
