import collections

import hdbscan
import numpy as np
import sklearn
import yaml
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt

from source import meaning, data, util
from source.fasttext import FastVector

config = yaml.load(open('homonyms.config'))


class WordCluster(object):
    def __init__(self, word, meaning_metric='average', clusterer='hdbscan', reduce_dimensions=False, normalize=False):
        if clusterer == 'hdbscan':
            self.clusterer = hdbscan.HDBSCAN(min_samples=int(data.get_counts()[word] * config['min_samples']),
                                             prediction_data=True)
        elif clusterer == 'kmeans':
            self.clusterer = KMeans(n_clusters=data.get_n_clusters()[word])
        elif clusterer == 'gmm':
            self.clusterer = GaussianMixture(n_components=data.get_n_clusters()[word])
        else:
            print('Unknown clusterer')
            exit(1)

        # Normalizing data if cosine distance not avail.
        self.clust_label = clusterer
        self.normalize = normalize
        self.data = data.get_data_for_word(word)
        self.word = word
        self.vectors = None
        self.meaning = meaning.get_meanings()[meaning_metric]
        self.reduce_dimensions = reduce_dimensions
        if reduce_dimensions:
            self.pca = PCA(n_components=config['reduce_dims'], whiten=True)

    def cluster(self):
        m = self.load_vectors(cache=True)
        if self.clust_label == 'gmm':
            self.clusterer.fit(m)
            self.labels = self.clusterer.predict(m)
        else:
            cluster_labels = self.clusterer.fit_predict(m)
            self.labels = cluster_labels
        self.counter = collections.Counter(self.labels)

    def load_vectors(self, cache=False):
        m = self.meaning(self.data, self.word)
        if self.reduce_dimensions:
            m = self.pca.fit_transform(m)
        if self.normalize:
            m = sklearn.preprocessing.normalize(m)
        if cache:
            self.vectors = m
        return m

    def load_ru_vectors(self, words):
        v = FastVector(vector_file=config['ru_vector'])
        v.apply_transform('./vec/ru.txt')
        m = np.vstack([v[word] for word in words])
        if self.reduce_dimensions:
            m = self.pca.transform(m)
        if self.normalize:
            m = sklearn.preprocessing.normalize(m)
        return m

    def print_cluster_overview(self):
        print('Word:{}'.format(self.word))
        print('Clusterer:{}'.format(self.clust_label))
        print('Number of clusters: {}'.format(len(self.counter.keys())))
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

    def test(self):
        tdat = data.get_test_data_for_word(self.word)
        ru_words = [sample[1] for sample in tdat]
        if self.word == 'bear':
            en_labels = self.labels[:20]
        else:
            en_labels = self.labels[:30]

        # None means Wera said toss this example. this is just foolery
        tmp = [(e, r) for e, r in zip(en_labels, ru_words) if r != 'none']
        en_labels = [i[0] for i in tmp]
        ru_words = [i[1] for i in tmp]

        vectors = self.load_ru_vectors(ru_words)
        if self.clust_label == 'hdbscan':
            ru_labels = hdbscan.approximate_predict(self.clusterer, vectors)[0]
        elif self.clust_label == 'kmeans' or 'gmm':
            ru_labels = self.clusterer.predict(vectors)
        else:
            print('Something went wrong')
            exit(1)

        return ru_labels, en_labels


if __name__ == '__main__':
    import random

    word = random.choice(data.get_word_list())
    WC = WordCluster(word, meaning_metric='average')

'''Artifacts

        # print([n for n in m if np.isnan(n)])
        # print([n for n in m if not np.isfinite(n)])
        # print(np.argwhere(np.isinf(m)))
'''
