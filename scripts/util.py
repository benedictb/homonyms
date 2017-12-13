#!/usr/bin/python3

import glob
import random
import string
import collections
import re
import traceback

caps = "([A-Z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"


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


# Probably should adjust this func so that it just spits out the sentence
def load_data_for_word(word):
    assert word in get_word_list()
    # data = []
    with open('./dat/preprocessed/{}.preprocessed'.format(word)) as f:
        data = clean_and_chop(f.readlines())
        # for sent in f:
        # data.append(clean([sent]))
    return data


def clean_and_chop(sents):
    table = str.maketrans(dict.fromkeys('''!"#$%&\\'()*+,./:;<=>?@[\\]^_`{|}~'''))
    res = []
    for sent in sents:
        cleaned = sent.strip('\n').lower().translate(table)
        chopped = re.split(' |-', cleaned)
        filtered = [w for w in chopped if w != '']
        res.append(filtered)
    return res


# For now, ignoring the passages with multiple occurrences in it
def load_test_data_for_word(word):
    assert word in get_word_list()

    with open('./dat/preprocessed/{}.preprocessed'.format(word)) as f:
        if word != 'bear':
            inp = [s for s in f.readlines()[:30]]
        else:
            inp = [s for s in f.readlines()[:20]]

    out = test_data_loader()[word]
    print(len(out))
    assert len(inp) == len(out)

    res = []
    for i, o in zip(inp, out):
        # None means Wera said toss this example
        if o == ['none']:
            continue
        try:
            sents = split_into_sentences(i)
            clean_sents = clean_and_chop(sents)
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


# Unweighted accuracy
def accuracy(l):
    return sum([1 if t[0] == t[1] else 0 for t in l]) / float(len(l))


# Pretty prints the result
def res_print(res):
    for pred, target in res:
        print('Pred:{} Target:{}'.format(pred[0], target))


# Target words used
def get_word_list():
    return ['bank', 'bat', 'bear', 'club', 'match', 'mess', 'mint', 'organ', 'stalk', 'volume']

# Taken from https://stackoverflow.com/questions/4576077/python-split-text-on-sentences
def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(caps + "[.]" + caps + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + caps + "[.]", " \\1<prd>", text)
    if "”" in text: text = text.replace(".”", "”.")
    if "\"" in text: text = text.replace(".\"", "\".")
    if "!" in text: text = text.replace("!\"", "\"!")
    if "?" in text: text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    if sentences:
        return sentences
    else:
        return [text]


if __name__ == '__main__':
    for w in get_word_list():
        print(w)
        print(load_test_data_for_word(w))
