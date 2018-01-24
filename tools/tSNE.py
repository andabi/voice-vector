# -*- coding: utf-8 -*-
#!/usr/bin/env python

#-*-encoding: utf-8 -*-
#!/usr/bin/python3

import numpy as np
import gensim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin', binary=True, limit=100000)
# model = gensim.models.word2vec.Word2Vec.load('glove.6B/glove.6B.50d.txt')
results = model.wv.similar_by_word('computer', topn=100)
words = ['computer']+[r[0] for r in results]
wordvectors = np.array(model['computer']+[model[w] for w in words], np.float32)
reduced = TSNE(n_components=2).fit_transform(wordvectors)
plt.figure(figsize=(20, 20))
max_x = np.amax(reduced, axis=0)[0]
max_y = np.amax(reduced, axis=0)[1]
plt.xlim((-max_x, max_x))
plt.ylim((-max_y, max_y))


plt.scatter(reduced[:, 0], reduced[:, 1], s=20, c=["r"] + ["b"]*(len(reduced)-1))

for i in range(100):
    target_word = words[i]
    print(target_word)
    x = reduced[i, 0]
    y = reduced[i, 1]
    plt.annotate(target_word, (x, y))

plt.savefig("glove_2000.png")
# plt.show()