#! /usr/bin/env python3

import util
import sys

try:
    lang = sys.argv[1]
except IndexError:
    lang = 'en'
if lang == 'en':
    en_alph = util.get_english_domain()
    f = open('./vec/fasttext/wiki.en.vec')
    for line in f:
        if line.split(' ')[0] in en_alph:
            print(line, end='')
elif lang == 'ru':
    ru_alph = util.get_english_domain()
    f = open('./vec/fasttext/wiki.ru.vec')
    for line in f:
        if line.split(' ')[0] in ru_alph:
            print(line, end='')
else:
    print('No lang found')
