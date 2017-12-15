import numpy as np
import yaml

from source.fasttext import FastVector

config = yaml.load(open('homonyms.config'))


def get_meanings():
    return {'average': average, 'semantics':semantics}


def average(data, word):
    model = FastVector(config['en_vector'])
    final_matrix = np.zeros([len(data), config['vector_length']])
    misses = 0
    for i, line in enumerate(data):
        line = [l for l in line if l != word]
        sent_vect = np.zeros([config['vector_length']])
        if not line:  # Sometimes very short sentences don't have anything left in them
            continue
        for j, word in enumerate(line):
            try:
                sent_vect += model[word]
            except KeyError as e:
                misses += 1
        ave = sent_vect / len(line)  # check this dimensionality here
        final_matrix[i] = ave

    print('Vector misses: {0} (Rate: {1:0.3f} words / sample)'.format(misses, (misses / len(data))))
    return final_matrix


def semantics(data, word,lang='en'):
    from polyglot.text import Text as T
    res = np.zeros([len(data)])
    for i, d in enumerate(data):
        try:
            d.remove(word)
            res[i] = T(' '.join(d), hint_language_code=lang).polarity
        except ZeroDivisionError:
            res[i] = 0.000001
        except ValueError:
            res[i] = 0.000001
        except AttributeError:
            res[i] = 0.000001

    # print('Semantics generated')
    return res.reshape(-1,1)