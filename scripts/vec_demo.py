def fasttext_demo():
    import fasttext
    model = fasttext.load_model('./vec/wiki.en.vec')
    vec = model['king']
    return vec


def glove_demo():
    import gensim
    return gensim.models.KeyedVectors.load_word2vec_format('./vec/w2v/glove.6B.{}d.w2vformat.txt'.format(str(50)),
                                                           binary=False)


model = glove_demo()
# model = fasttext_demo()

while True:
    try:
        word = input('>')
        print(model.most_similar(positive=word))
    except Exception:
        pass
