#!/usr/bin/python3

import glob
import string
import collections

# Returns set of all english words in data
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
    with open('./dat/test/test.txt') as f:
        items = [i.strip('\n').lower().split(' ')[1:] for i in f]
        for i in items:
            words.update(i)
    words.discard('')
    return words

# Returns a dictionary of english word : list of english sentences (list form) containing that word
def train_data_loader(filepath='./dat/preprocessed/', randomize=False):
    d = {}
    for f in glob.glob(filepath + '*.preprocessed'):
        word = f.split('/')[-1].split('.')[0]
        d[word] = []
        for sent in open(f).readlines():
            table = str.maketrans(dict.fromkeys(string.punctuation))
            psent = [w for w in sent.strip('\n').lower().translate(table).split(' ') if w != '']
            d[word].append(psent)
    return d

# Returns dictionary of english word : list of russian translations
def test_data_loader(filepath='./dat/test/test.txt', randomize=False):
    lines = [l.strip('\n').lower() for l in open(filepath)]
    data = [(l.split(' ')[0], l.split(' ')[1:]) for l in lines]
    if randomize:
   	    random.shuffle(data)
    d = collections.defaultdict(list)
    for l in data:
        d[l[0]].append(l[1])
    return d

def res_print(res):
    for pred, target in res:
        print('Pred:{} Target:{}'.format(pred[0], target))

    print('\n\nACC:{}'.format(accuracy(res)))

def get_word_list():
    return ['club', 'bank', 'bat', 'bear', 'club', 'match', 'mess', 'mint', 'organ', 'stalk', 'volume']


if __name__ == '__main__':
    pass