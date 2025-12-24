'''NLP Tutorial: Word Vectors Overview Using Gensim Library
All gensim models are listed on this page: https://github.com/RaRe-Technologies/gensim-data'''

import gensim.downloader as api
# This is a huge model (~1.6 gb) and it will take some time to load

wv = api.load('word2vec-google-news-300') #It is a trained model using Google New 

print(wv.similarity(w1="great", w2="good"))

print(wv.most_similar("good"))

print(wv.most_similar("dog"))

print(wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=5))

print(wv.most_similar(positive=['france', 'berlin'], negative=['paris'], topn=5))

print(wv.doesnt_match(["dog", "cat", "google", "mouse"]))

print(wv.doesnt_match(["facebook", "cat", "google", "microsoft"]))

'''Gensim: Glove
Stanford's page on GloVe: https://nlp.stanford.edu/projects/glove/'''

glv = api.load("glove-twitter-25") #model trained based on Twitter data.
glv.most_similar("good")

print(glv.doesnt_match("breakfast cereal dinner lunch".split()))

print(glv.doesnt_match("facebook cat google microsoft".split()))

print(glv.doesnt_match("banana grapes orange human".split()))