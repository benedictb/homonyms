# Returns set of all english words in data
import collections
import glob
import random
import string

import source.util as util
import yaml

from source.fasttext import FastVector

config = yaml.load(open('homonyms.config'))


# Returns set of all english words in parallel data (not all data)
def get_english_domain():
    words = set()
    dir_path = './dat/preprocessed/'
    for f in glob.glob(dir_path + '*.preprocessed'):
        for sent in open(f).readlines():
            table = str.maketrans(dict.fromkeys(string.punctuation))
            psent = sent.strip('\n').lower().translate(table).split(' ')
            words.update(psent)
    words.discard('')
    return words


# Returns set of all russian words in data
def get_russian_domain():
    words = set()
    with open(config['testfile']) as f:
        items = [i.strip('\n').lower().split(' ')[1:] for i in f]
        for i in items:
            words.update(i)
    words.discard('')
    return words


def get_reduced_russian_domain():
    words = test_data_loader()
    s = set()
    for k in words.keys():
        russian_words_for_this_homonym = [l[0] for l in words[k]]
        s.update(russian_words_for_this_homonym)
    s.discard('none')
    return s


# Probably should adjust this func so that it just spits out the sentence
def get_data_for_word(word, limit=None):
    assert word in get_word_list()
    with open('./dat/filtered/{}.dat'.format(word)) as f:
        # Turn into list
        if limit:
            chopped = util.clean_and_chop(f.readlines()[:limit])
        else:
            chopped = util.clean_and_chop(f.readlines())

        # Filter out stops
        filtered = filterer(chopped)

        # Trim around the word
        trimmed = util.trim(filtered, word, trim=config['trim'])

    return trimmed


# For now, ignoring the passages with multiple occurrences in it
def get_test_data_for_word(word):
    assert word in get_word_list()

    # with open('./dat/preprocessed/{}.preprocessed'.format(word)) as f:
    if word == 'bear':
        inp = get_data_for_word(word, limit=20)
    else:
        inp = get_data_for_word(word, limit=30)

    out = test_data_loader()[word]
    assert len(inp) == len(out)

    res = []
    for i, o in zip(inp, out):
        res.append((i, o[0]))
    return res


# Returns dictionary of english word : list of russian translations
def test_data_loader(filepath=config['testfile'], randomize=False):
    lines = [l.strip('\n').lower() for l in open(filepath)]
    data = [(l.split(' ')[0], l.split(' ')[1:]) for l in lines]
    if randomize:
        random.shuffle(data)
    d = collections.defaultdict(list)
    for l in data:
        d[l[0]].append(l[1])
    return d


# Target words used, alphabetical order because why not
def get_word_list():
    return ['bank', 'bat', 'bear', 'club', 'match', 'mess', 'mint', 'organ', 'stalk', 'volume']


def filterer(sents):
    stops = util.get_stop_words()
    res = []
    for sent in sents:
        # filter out stop words
        sent = [w for w in sent if w not in stops]

        # filter out numbers. Not the most efficient thing on the block
        # sent = ['number' if any(i.isdigit() for i in w) else w for w in sent]
        sent = [w for w in sent if not any(i.isdigit() for i in w)]

        res.append(sent)
    return res


# Estimation of num clusters, this would be better if I was better at statistics
# But this is a way to get a approximate amount of clusters for each word without changing it every time
def get_n_clusters():
    counts = []
    testData = test_data_loader()
    words = list(testData.keys())
    for k in words:
        count = len(set([l[0] for l in testData[k]]))
        counts.append(count)

    mx = max(counts)
    mn = min(counts)

    n_counts = [(c - mn) / (mx - mn) for c in counts]

    max_cluster = config['max_clusters']
    min_cluster = config['min_clusters']
    diff = max_cluster - min_cluster

    clusters = [min_cluster + int(i * diff) for i in n_counts]

    return {k: v for k, v in zip(words, clusters)}


def get_counts():
    return {'bank': 169117, 'club': 108071, 'match': 96666, 'bear': 22020, 'volume': 20683, 'mess': 14100, 'bat': 9524,
            'organ': 5988, 'mint': 1523, 'stalk': 616}


if __name__ == '__main__':

    v = FastVector(vector_file=config['ru_vector'])
    v.apply_transform('./vec/ru.txt')

    s = get_reduced_russian_domain()

    # I need to stop using one letter variable names, I'm running out
    for russian_word in s:
        try:
            _ = v[russian_word]
        except KeyError:
            print(russian_word)