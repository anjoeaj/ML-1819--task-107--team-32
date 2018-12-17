import pandas as pd

import os
import nltk
import gensim
from gensim import corpora, models

df = pd.read_csv('words_dataset.csv', delimiter=",")

tokenized_text = [nltk.word_tokenize(sent) for sent in df['text']]

# vary the size parameter and check the results
# higher dimension of vector lead to overfit, vary between 32 to 55
model = gensim.models.Word2Vec(tokenized_text,
                               min_count=1, size=50)

# save
# model.save("word2vec")

# train
