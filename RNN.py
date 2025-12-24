
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re 

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd 

paragrah ='''NLTK is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries, and an active discussion forum.

Thanks to a hands-on guide introducing programming fundamentals alongside topics in computational linguistics, plus comprehensive API documentation, NLTK is suitable for linguists, engineers, students, educators, researchers, and industry users alike. NLTK is available for Windows, Mac OS X, and Linux. Best of all, NLTK is a free, open source, community-driven project.

NLTK has been called “a wonderful tool for teaching, and working in, computational linguistics using Python,” and “an amazing library to play with natural language.”

'''
#tokenization convert paragrah -sentences-words
nltk.download('punkt')
sentences = nltk.sent_tokenize(paragrah)
print(sentences)
stemmer = PorterStemmer()
print(stemmer.stem('Going'))#base words it will cut no meaning some time.
lemmatizer = WordNetLemmatizer() 
print(lemmatizer.lemmatize('Going')) #it will create meaning ful words.

#cleaning the text 
corpus =[]
for i in range(len(sentences)) :
    text = re.sub('[^a-zA-Z]',' ',sentences[i]) # Other than a-zA-Z replace with empty string '' in sentences.
    text = text.lower()
    text = text.split()
    text  = [lemmatizer.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]
    text  = " ".join(text)
    corpus.append(text)
print(corpus)

#stemming 

'''Reduce a word to its root form by cutting off prefixes or suffixes.
It doesn’t care if the result is a real English word — it’s 
fast but not linguistically accurate.'''

stemmingWords =[]
for i in corpus:
    words = nltk.word_tokenize(i)
    for word in words:
        if word not in set(stopwords.words('english')):
            stemmingWords.append(stemmer.stem(word))
print('words after stemming:\n',stemmingWords)

#Lemmatization — smart normalization
'''Reduce words to their dictionary (lemma) form, considering grammar and context.
It uses the WordNet lexical database, so results are proper English words.'''
wordLemmatizer =[]
for i in corpus:
    words = nltk.word_tokenize(i)
    for word in words:
        if word not in set(stopwords.words('english')):
            wordLemmatizer.append(lemmatizer.lemmatize(word))
print('words after WordNetLemmatizer:\n',wordLemmatizer)

cv =  CountVectorizer()
X = cv.fit_transform(corpus)

print(cv.vocabulary_)

print('corpus of zero:\n',corpus[0])
print(X[0].shape)

from transformers import pipeline

text = "Apple Inc. announced record profits this quarter, driven by strong iPhone sales in Asia."

# Sentiment
sentiment = pipeline("sentiment-analysis")
print(sentiment(text))

# Named Entity Recognition
ner = pipeline("ner", grouped_entities=True)
print(ner(text))

# Summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print(summarizer(text, max_length=30, min_length=5, do_sample=False))

# Text Classification (topic)
classifier = pipeline("zero-shot-classification")
labels = ["business", "sports", "technology", "politics"]
print(classifier(text, candidate_labels=labels))

