import collections

import hdbscan
import sklearn

from scripts import meaning, data, util

min_samples = 10


class WordCluster(object):
    def __init__(self, word, meaning_metric='average'):
        self.data = data.get_data_for_word(word)
        self.word = word
        self.clusterer = hdbscan.HDBSCAN(min_samples=min_samples)
        self.vectors = None
        self.meaning = meaning.get_meanings()[meaning_metric]

    def cluster(self):
        m = self.load_vectors(cache=True)
        cluster_labels = self.clusterer.fit_predict(m)
        self.labels = cluster_labels
        self.counter = collections.Counter(self.labels)

    def load_vectors(self, cache=False):
        m = self.meaning(self.data, self.word)
        if cache:
            self.vectors = m
        return m

    def print_clusters(self):
        print('Number of clusters: {}'.format(len(self.counter.keys()) - 1))
        for k in self.counter.keys():
            if k >= 0:
                print('Cluster {}: {} samples'.format(k, self.counter[k]))
        print('Unlabeled: {} samples'.format(self.counter[-1]))

    def print_cluster_data(self, cluster_num):
        print('Samples in cluster {}:'.format(cluster_num))
        indices = [i for i in range(len(self.labels)) if self.labels[i] == cluster_num]
        for index in indices:
            print(' '.join([w.upper() if w == self.word else w for w in self.data[index]]))

    def explore(self):
        while True:
            self.print_clusters()
            inp = int(input('Select a key>'))
            self.print_cluster_data(cluster_num=inp)


if __name__ == '__main__':
    import random

    word = random.choice(data.get_word_list())
    WC = WordCluster(word, meaning_metric='average')
