# Text Classification USing Gensim Word Embeddings

#Load Google News Word2vec model from gensim library

import gensim.downloader as api
import pandas as pd
from sklearn.model_selection  import train_test_split
from sklearn.metrics import classification_report
import spacy
import numpy as np 


wv = api.load('word2vec-google-news-300')
wv.similarity(w1="great", w2="good")

wv_great = wv["great"] #It will create word to vector means numbers
wv_good = wv["good"]
print(wv_great)
print(wv_good) 

print('Shape of the vectors are:\n')
print(wv_great.shape,wv_good.shape)

'''Fake vs Real News Classification Using This Word2Vec Embeddings
Fake news refers to misinformation or disinformation in the country which is spread through word of mouth and more recently through digital 
communication such as What's app messages, social media posts, etc.

Fake news spreads faster than real news and creates problems and fear among groups and in society.

We are going to address these problems using classical NLP techniques and going to classify whether a given message/ text is Real or Fake Message.

We will use glove embeddings from spacy which is trained on massive wikipedia dataset to pre-process and text 
vectorization and apply different classification algorithms.

Dataset
Credits: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

This data consists of two columns. - Text - label

Text is the statements or messages regarding a particular event/situation.

label feature tells whether the given text is Fake or Real.

As there are only 2 classes, this problem comes under the Binary Classification.'''

df = pd.read_csv('C:\ML\DataSet\\fake_and_real_news.csv')

print(df[:10])

df['Label_num'] = df['label'].map({'Real':1,"Fake":0})

print(df[:10])

nlp = spacy.load('en_core_web_lg')

#Now let's write the function that can do preprocessing and vectorization both
filtered_token = []
def preprocess_vectorize(text):
    doc = nlp(text)
    for token in doc:
        if token.is_stop or token.is_punct:
              continue
        filtered_token.append(token.lemma_)
    return wv.get_mean_vector(filtered_token) 


'''Now we will convert the text into a vector using gensim's word2vec embeddings.
We will do this in three steps,
Preprocess the text to remove stop words, punctuations and get lemma for each word
Get word vectors for each of the words in a pre-processed sentece
Take a mean of all word vectors to derive the numeric representation of the entire news article
First let's explore get_mean_vector api of gensim to see how it works

r1 = np.mean([wv_good, wv_great],axis=0)
'''
r1 = np.mean([wv_good, wv_great],axis=0)
wv_good[:5]

r2 = wv.get_mean_vector(["good", "great"],pre_normalize=False)

v = preprocess_vectorize("Don't worry if you don't understand")
print(v.shape)

df['vector'] = df['Text'].apply(lambda x : preprocess_vectorize(x))

from sklearn.model_selection import train_test_split


#Do the 'train-test' splitting with test size of 20% with random state of 2022 and stratify sampling too
X_train, X_test, y_train, y_test = train_test_split(
    df.vector.values, 
    df.label_num, 
    test_size=0.2, # 20% samples will go to test dataset
    random_state=2022,
    stratify=df.label_num
)
#Reshaping the X_train and X_test so as to fit for models

print("Shape of X_train before reshaping: ", X_train.shape)
print("Shape of X_test before reshaping: ", X_test.shape)


X_train_2d = np.stack(X_train) #changing the array of array to 2d array so that we can feed to model
X_test_2d =  np.stack(X_test)

print("Shape of X_train after reshaping: ", X_train_2d.shape)
print("Shape of X_test after reshaping: ", X_test_2d.shape)

#I tried Random forest, decision tree, naive bayes etc classifiers as well but gradient boosting gave the best performance of all

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

#1. creating a GradientBoosting model object
clf = GradientBoostingClassifier()

#2. fit with all_train_embeddings and y_train
clf.fit(X_train_2d, y_train)


#3. get the predictions for all_test_embeddings and store it in y_pred
y_pred = clf.predict(X_test_2d)


#4. print the classfication report
print(classification_report(y_test, y_pred))

#Make some predictions

test_news = [
    "Michigan governor denies misleading U.S. House on Flint water (Reuters) - Michigan Governor Rick Snyder denied Thursday that he had misled a U.S. House of Representatives committee last year over testimony on Flintâ€™s water crisis after lawmakers asked if his testimony had been contradicted by a witness in a court hearing. The House Oversight and Government Reform Committee wrote Snyder earlier Thursday asking him about published reports that one of his aides, Harvey Hollins, testified in a court hearing last week in Michigan that he had notified Snyder of an outbreak of Legionnairesâ€™ disease linked to the Flint water crisis in December 2015, rather than 2016 as Snyder had testified. â€œMy testimony was truthful and I stand by it,â€ Snyder told the committee in a letter, adding that his office has provided tens of thousands of pages of records to the committee and would continue to cooperate fully.  Last week, prosecutors in Michigan said Dr. Eden Wells, the stateâ€™s chief medical executive who already faced lesser charges, would become the sixth current or former official to face involuntary manslaughter charges in connection with the crisis. The charges stem from more than 80 cases of Legionnairesâ€™ disease and at least 12 deaths that were believed to be linked to the water in Flint after the city switched its source from Lake Huron to the Flint River in April 2014. Wells was among six current and former Michigan and Flint officials charged in June. The other five, including Michigan Health and Human Services Director Nick Lyon, were charged at the time with involuntary manslaughter",
    " WATCH: Fox News Host Loses Her Sh*t, Says Investigating Russia For Hacking Our Election Is Unpatriotic This woman is insane.In an incredibly disrespectful rant against President Obama and anyone else who supports investigating Russian interference in our election, Fox News host Jeanine Pirro said that anybody who is against Donald Trump is anti-American. Look, it s time to take sides,  she began.",
    " Sarah Palin Celebrates After White Man Who Pulled Gun On Black Protesters Goes Unpunished (VIDEO) Sarah Palin, one of the nigh-innumerable  deplorables  in Donald Trump s  basket,  almost outdid herself in terms of horribleness on Friday."
]

test_news_vectors = [preprocess_vectorize(n) for n in test_news]
clf.predict(test_news_vectors)

#Confusion Matrix for Best Model

#finally print the confusion matrix for the best model (GradientBoostingClassifier)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


from matplotlib import pyplot as plt
import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Prediction')
plt.ylabel('Truth')

