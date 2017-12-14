import collections

import hdbscan
import numpy as np
import sklearn
import yaml
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from scripts import meaning, data, util

config = yaml.load(open('homonyms.config'))


class WordCluster(object):
    def __init__(self, word, meaning_metric='average', clusterer='hdbscan', reduce_dimensions=False):
        if clusterer == 'hdbscan':
            self.clusterer = hdbscan.HDBSCAN(min_samples=config['min_samples'])
        elif clusterer == 'kmeans':
            self.clusterer = KMeans(n_clusters=config['n_clisters'])
        elif clusterer == 'agglomerative':
            self.clusterer = AgglomerativeClustering(n_clusters=config['min_samples'], affinity='cosine', linkage='average')

        self.data = data.get_data_for_word(word)
        self.word = word
        self.vectors = None
        self.meaning = meaning.get_meanings()[meaning_metric]
        self.reduce_dimensions = reduce_dimensions
        if reduce_dimensions:
            self.pca = PCA(n_components=reduce_dimensions, whiten=True)

    def cluster(self):
        m = self.load_vectors(cache=True)
        cluster_labels = self.clusterer.fit_predict(m)
        self.labels = cluster_labels
        self.counter = collections.Counter(self.labels)

    def load_vectors(self, cache=False):
        m = self.meaning(self.data, self.word)
        #print([n for n in m if np.isnan(n)])
        #print([n for n in m if not np.isfinite(n)])
        print(np.argwhere(np.isinf(m)))
        if self.reduce_dimensions:
            m = self.pca.fit_transform(m)
        if cache:
            self.vectors = m
        return m

    def print_cluster_overview(self):
        print('Number of clusters: {}'.format(len(self.counter.keys()) - 1))
        for k in self.counter.keys():
            if k >= 0:
                print('Cluster {}: {} samples'.format(k, self.counter[k]))
        print('Unlabeled: {} samples'.format(self.counter[-1]))

    def print_cluster(self, cluster_num):
        print('Samples in cluster {}:'.format(cluster_num))
        indices = [i for i in range(len(self.labels)) if self.labels[i] == cluster_num]
        for index in indices:
            print('> ' + ' '.join([w.upper() if w == self.word else w for w in self.data[index]]))

    def explore(self):
        while True:
            self.print_cluster_overview()
            inp = input('Select a key>')
            if inp == 'q':
                return
            else:
                self.print_cluster(cluster_num=int(inp))

    def visualize(self):
        self.print_cluster_overview()
        tsne = TSNE(n_components=2, verbose=1)
        res = tsne.fit_transform(self.vectors)
        vis_x = res[:, 0]
        vis_y = res[:, 1]
        plt.scatter(vis_x, vis_y, c=self.labels, alpha=.1)
        plt.show()

        # print(res.shape)


if __name__ == '__main__':
    import random

    word = random.choice(data.get_word_list())
    WC = WordCluster(word, meaning_metric='average')
