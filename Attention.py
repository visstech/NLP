import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import string
import numpy as np 
import pandas as pd 

Sample_text ="""Word2vec  popular technique natural language processing.""" 
#"""It convert words into numerical vectors such that similar words have similar vector
#representation. Word2vec is trained on large corpora of text and learns words embeddings 
#using a neural network model"""

tokens = word_tokenize(Sample_text.lower())
tokens = [word for word in tokens if word not in string.punctuation]
model  = Word2Vec([tokens],vector_size=2,window=5,min_count=1)
word_embedding = {word:model.wv[word] for word in model.wv.index_to_key}
words = list(word_embedding.keys())
embeddings = np.array([word_embedding[word]for word in words])
Q = embeddings
K = embeddings
V = embeddings
scores = np.matmul(Q,K.T)
d_k = Q.shape[-1]
print(d_k)
scores = scores /d_k**0.5
e_x = np.exp(scores - np.max(scores,axis=1,keepdims=True))
attention_score = e_x /np.sum(e_x,axis=1,keepdims=True)
output = np.matmul(attention_score,V)
df = pd.DataFrame(attention_score,columns=words,index=words)
print(df)