# Returns set of all english words in data
import collections
import glob
import random
import string
import traceback

import scripts.util as util


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
    with open('./dat/test/test.txt') as f:
        items = [i.strip('\n').lower().split(' ')[1:] for i in f]
        for i in items:
            words.update(i)
    words.discard('')
    return words


# Probably should adjust this func so that it just spits out the sentence
def get_data_for_word(word):
    assert word in get_word_list()
    with open('./dat/filtered/{}.dat'.format(word)) as f:
        chopped =  util.clean_and_chop(f.readlines())
        filtered = filterer(chopped)
    return filtered


# For now, ignoring the passages with multiple occurrences in it
def get_test_data_for_word(word):
    assert word in get_word_list()

    with open('./dat/preprocessed/{}.preprocessed'.format(word)) as f:
        if word != 'bear':
            inp = [s for s in f.readlines()[:30]]
        else:
            inp = [s for s in f.readlines()[:20]]

    out = test_data_loader()[word]
    assert len(inp) == len(out)

    res = []
    for i, o in zip(inp, out):
        # None means Wera said toss this example
        if o == ['none']:
            continue
        try:
            sents = util.split_into_sentences(i)
            clean_sents = util.clean_and_chop(sents)
            contains = [sent for sent in clean_sents if word in sent]
            res.append((contains[0], o[0]))

            # This is tricky because sometimes a sentence has a word twice. Ya know
            # for j, target in enumerate(o):
            #     res.append((contains[j], target))
        except IndexError:
            traceback.print_exc()
            print('i:{}'.format(i))
            print('o:{}'.format(o))
            print('sents:{}'.format(sents))
            exit(0)
    return res


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


# Target words used
def get_word_list():
    return ['bank', 'bat', 'bear', 'club', 'match', 'mess', 'mint', 'organ', 'stalk', 'volume']


def filterer(sents):
    stops = util.get_stop_words()
    res = []
    for sent in sents:
        # filter out stop words
        sent = [w for w in sent if w not in stops]

        # filter out numbers. Not the most efficient thing on the block
        sent = ['number' if any(i.isdigit() for i in w) else w for w in sent]

        res.append(sent)
    return res


if __name__ == '__main__':
    for w in get_word_list():
        print(w)
        print(get_test_data_for_word(w))

'''
Artifacts

# Returns a dictionary of english word : list of english sentences (list form) containing that word
# def train_data_loader(filepath='./dat/preprocessed/', randomize=False):
#     d = {}
#     for f in glob.glob(filepath + '*.preprocessed'):
#         word = f.split('/')[-1].split('.')[0]
#         d[word] = []
#         for sent in open(f).readlines():
#             table = str.maketrans(dict.fromkeys(string.punctuation))
#             psent = [w for w in sent.strip('\n').lower().translate(table).split(' ') if w != '']
#             d[word].append(psent)
#     return d

'''
