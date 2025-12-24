'''Text Classification Using Spacy Word Embeddings
Problem Statement
Fake news refers to misinformation or disinformation in the country which is spread through word of mouth and more recently through digital communication such as What's app messages, social media posts, etc.

Fake news spreads faster than real news and creates problems and fear among groups and in society.

We are going to address these problems using classical NLP techniques and going to classify whether a given message/ text is Real or Fake Message.

We will use glove embeddings from spacy which is trained on massive wikipedia dataset to pre-process and text vectorization and apply different classification algorithms.

Dataset
Credits: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

This data consists of two columns. - Text - label

Text is the statements or messages regarding a particular event/situation.

label feature tells whether the given text is Fake or Real.

As there are only 2 classes, this problem comes under the Binary Classification.'''
import spacy
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


nlp = spacy.load('en_core_web_lg')
df = pd.read_csv('C:\\ML\\DataSet\\Fake_Real_Data.csv')

print(df[:10])
df['label_num'] = df['label'].map({"Fake":0,"Real":1})

print(df[:10])

print(df.shape)
df= df[:1000]
df['Text_vector'] = df['Text'].apply(lambda x: nlp(x).vector)
scaler = MinMaxScaler()

X_train,X_test,y_train,y_test = train_test_split(df['Text_vector'],df['label_num'],test_size=0.2,random_state=101)

import numpy as np

X_train_2d = np.stack(X_train)
X_test_2d = np.stack(X_test)
X_train_scaled = scaler.fit_transform(X_train_2d)
X_test_scaled = scaler.fit_transform(X_test_2d)

clf = MultinomialNB()
clf.fit(X_train_scaled, y_train)
y_predicted = clf.predict(X_test_scaled)
print('Classification report for MultinomialNB Performance metrics are:\n',classification_report(y_test, y_predicted))


#1. creating a KNN model object
clf = KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean')

#2. fit with all_train_embeddings and y_train
clf.fit(X_train_2d, y_train)

#3. get the predictions for all_test_embeddings and store it in y_pred
y_pred = clf.predict(X_test_2d)

#4. print the classfication report
print('Classification report for KNeighborsClassifier Performance metrics are:\n',classification_report(y_test, y_pred))


#Confusion Matrix

#finally print the confusion matrix for the best model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

from matplotlib import pyplot as plt
import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Prediction')
plt.ylabel('Truth')
plt.show()

'''Key Takeaways
KNN model which didn't perform well in the vectorization techniques like Bag of words, and TF-IDF due to very high 
dimensional vector space, performed really well with glove vectors due to only 300-dimensional vectors and very good 
embeddings(similar and related words have almost similar embeddings) for the given text data.

MultinomialNB model performed decently well but did not come into the top list because in the 300-dimensional
vectors we also have the negative values present. The Naive Bayes model does not fit the data if there are negative values. 
So, to overcome this shortcoming, we have used the Min-Max scaler to bring down all the values between 0 to 1. In this process,
there will be a possibility of variance and information loss among the data. But anyhow we got a decent recall and f1 scores.'''