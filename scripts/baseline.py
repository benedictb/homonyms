#! /usr/bin/env python

# bbecker5

'''

Baseline Plan (took a bit longer than an hour):

For given input word:
- Find all lines in db that have that word
- For each sentence, calculate the aggregate positive, negative sentiment. Normalize both between 0,1 to account for length of sentence.
- Run k-means clustering on the result. Train models from k = 1 to say 5. Pick the best using some heuristic (tbd)
- Then, for each cluster, find the sentence with the smallest Euclidean distance to it's center (in this case, using its positive and negative values).
    - Perhaps two would be better to get a better idea 
- Display those sentences, hopefully it will be clear what each cluster represents for obvious words


Baseline Results

Don't currently have enough data for meaningful results. An upshot of this is that the models are defined enough to create an accurate heuristic for 
determining the best number of clusters. Right now, I have about 10k English entries, but can easily get access to a 15 million character english db. 

'''

import math
import string
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from junk.data_loader import load_data

SENTIMENT_PATH = './dat/sentiment/sentiwordnet.txt'
sent = pd.read_table(SENTIMENT_PATH, usecols=[2, 3, 4], index_col=['SynsetTerms'])


def get_sent(line):
    pos = 0
    neg = 0
    for w in line.lower().split(' '):
        try:
            r = sent.loc[w]  # This is a little bit wack
            pos += r[0]
            neg += r[1]
        except KeyError:
            pass

    # Need to normalize these somehow
    return [pos, neg]


def parse(cur, word):
    res = dict()
    for rec in cur.fetchall():
        for line in rec[1].split('.')[:-1]:  # Sometimes junk on the end in the corpus I'm using
            if word in line:
                res[rec[0]] = get_sent(line.replace(word, ''))  # Don't want the word influencing it up
                break
    return res

def masc_parse(word, data, index):
    try:
        sents = [data[line] for line in index[word]]
        print(sents)
    except KeyError:
        print('"{}" not found in corpus'.format(word))
    res = dict()
    for sent in sents:
        cleaned = ' '.join([tok.translate(None, string.punctuation) for tok in line.lower().split(' ') if tok != word])
        res[sent] = get_sent(cleaned)
    return res

def cluster(d):
    l = [(k, np.asarray(d[k])) for k in d.keys()]
    m = np.asarray([np.asarray(i[1]) for i in l if i[1][0] or i[1][1]])  # Make sure there's something there
    scores = [0] * 5
    for k in range(5):
        kmeans = KMeans(n_clusters=k + 1, random_state=0).fit(m)
        scores[k] = kmeans.score(m)

    plt.plot(range(5), scores, label='linear')
    plt.show()

    final_k = 3
    # Some metric for picking the model, maybe distance between centers of clusters.


    model = KMeans(n_clusters=final_k, random_state=0).fit(m)
    best_matches = []
    for cluster in model.cluster_centers_:
        lowest = min(l, key=lambda x: euc_dis(cluster, x[1]))[0]
        best_matches.append(lowest)
    return best_matches, final_k


def euc_dis(a1, a2):
    return math.sqrt(math.pow(a1[0] - a2[0], 2) + math.pow(a1[1] - a2[1], 2))


def baseline(word):
    # db = MySQLdb.connect(host="localhost",  # your host, usually localhost
    #                      user="root",  # your username
    #                      passwd="password",  # your password
    #                      db="omega")  # name of the data base
    # cur = db.cursor()
    #
    # query = "SELECT t.text_id as id, t.text_text as text from uw_text t, uw_translated_content c where t.text_id = c.text_id and c.language_id = 85 and t.text_text like '% {} %'".format(
    #     word)
    # cur.execute(query)

    data, index = load_data()
    sentiments = masc_parse(word, data, index)
    best_fits, k = cluster(sentiments)

    print('Target word:	{}'.format(word))
    print('Optimal model:	{}'.format(k))
    print('Best Fits:')

    print(best_fits)

    # for i, fit in enumerate(best_fits):
        # cur.execute('SELECT text_text FROM uw_text WHERE text_id = {}'.format(fit))
        # text = cur.fetchall()

        # for t in text:
        #     print 'Cluster {}: '.format(i) + t[0]


if __name__ == '__main__':
    word = sys.argv[1]
    baseline(word)
