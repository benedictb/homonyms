#!/usr/bin/python3
import re

import sklearn

caps = "([A-Z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"


def clean_and_chop(sents):
    table = str.maketrans(dict.fromkeys('''!"#$%&\\'()*+,./:;<=>?@[\\]^_`{|}~'''))
    res = []
    for sent in sents:
        cleaned = sent.strip('\n').lower().translate(table)
        chopped = re.split(' |-|—', cleaned)
        filtered = [w for w in chopped if w != '']
        res.append(filtered)
    return res


# Unweighted accuracy
def accuracy(pred, truth):
    return sum([t == p for t, p in zip(truth, pred)]) / float(len(truth))


def w_f1(pred, truth):
    return sklearn.metrics.f1_score(truth, pred, average='weighted')


# Pretty prints the result
def res_print(res):
    for pred, target in res:
        print('Pred:{} Target:{}'.format(pred, target))


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


# https://algs4.cs.princeton.edu/35applications/stopwords.txt
def get_stop_words():
    with open('./dat/stopwords.txt') as f:
        return [w.strip('\n').replace(',', '') for w in f]


def trim(sents, word, trim=7):
    res = []
    for s in sents:
        try:
            idx = s.index(word)
            res.append(s[max(0, idx - trim):min(len(s), idx + trim)])
        except ValueError:
            # print(s)
            continue
            # exit(0)
    return res


if __name__ == '__main__':
    pass
