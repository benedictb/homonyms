#! /usr/bin/env python3
import random

from source.word_cluster import WordCluster
from source import util, data

'''try gmm'''

if __name__ == '__main__':

    # # This is pretty inefficient because it loads the word vectors from memory every time
    # list = data.get_word_list()
    # random.shuffle(list)
    # for word in list:
    #     print('word:{}'.format(word))
    #     s = 0
    #     for i in range(5):
    #         print('iter {}'.format(i))
    #         wc = WordCluster(word=word, clusterer='kmeans', reduce_dimensions=True, normalize=True)
    #         wc.cluster()
    #         wc.print_cluster_overview()
    #         pred, truth = wc.test()
    #         f1 = util.w_f1(pred, truth)
    #         # print(acc)
    #         s += f1
    #     print('F1:{}'.format(s / float(5)))

    word = random.choice(data.get_word_list())
    print('word:{}'.format(word))
    s = 0
    for i in range(5):
        print('iter {}'.format(i))
        wc = WordCluster(word=word, clusterer='gmm', reduce_dimensions=True, normalize=True)
        wc.cluster()
        wc.print_cluster_overview()
        pred, truth = wc.test()
        f1 = util.w_f1(pred, truth)
        print(f1)
        # if f1 > .9: # want a good visual
        #     wc.visualize()
        s += f1
    print('F1:{}'.format(s / float(5)))

