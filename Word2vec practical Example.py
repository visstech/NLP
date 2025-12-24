#Word2vec practical Example
import gensim
from gensim.models import word2vec,KeyedVectors
import gensim.downloader as api
wv = api.load('word2vec-google-news-300')
vec_king =wv['king']
print(vec_king)
print(wv.most_similar('Senthil'))
print(wv.similarity(['senthil','kumar','w2']))