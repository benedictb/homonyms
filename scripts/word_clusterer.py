import numpy as np
import sklearn
from scripts import meaning
from scripts.data_loader import get_word_list, load_data_for_word


class WordCluster(object):
    def __init__(self, word, meaning_metric='average'):
        self.data = load_data_for_word(word)
        self.word = word
        self.clusterer = sklearn.cluster.DBSCAN()
        self.vectors = None
        if meaning_metric == 'average':
            self.meaning = meaning.average

    def cluster(self):
        # for matrix
        m = self.load_vectors(cache=True)


    def load_vectors(self, cache=False):
        m = self.meaning(self.data, self.word)
        if cache:
            self.vectors = m
        return m

if __name__ == '__main__':
    import random
    word = random.choice(get_word_list())
    WC = WordCluster(word,meaning_metric='average')