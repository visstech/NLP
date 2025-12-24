'''What is TF-IDF?
TF stands for Term Frequency and denotes the ratio of number of times a 
particular word appeared in a Document to total number of words in the document.

   Term Frequency(TF) = [number of times word appeared / total no of words in a document]
Term Frequency values ranges between 0 and 1. If a word occurs more number of times,
then it's value will be close to 1.

IDF stands for Inverse Document Frequency and denotes the log of ratio of 
total number of documents/datapoints in the whole dataset to the number of 
documents that contains the particular word.

   Inverse Document Frequency(IDF) = [log(Total number of documents / number of 
   documents that contains the word)]
In IDF, if a word occured in more number of documents and is common across all 
documents, then it's value will be less and ratio will approaches to 0.

Finally:

   TF-IDF = Term Frequency(TF) * Inverse Document Frequency(IDF)'''
   
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB 
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import spacy
from sklearn.metrics import confusion_matrix

corpus = [
    "Thor eating pizza, Loki is eating pizza, Ironman ate pizza already",
    "Apple is announcing new iphone tomorrow",
    "Tesla is announcing new model-3 tomorrow",
    "Google is announcing new pixel-6 tomorrow",
    "Microsoft is announcing new surface tomorrow",
    "Amazon is announcing new eco-dot tomorrow",
    "I am eating biryani and you are eating grapes"
]

TFIDF =TfidfVectorizer()
Transformed_output = TFIDF.fit_transform(corpus)
print(Transformed_output.toarray())

#let's print the vocabulary

print(TFIDF.vocabulary_)

#let's print the idf of each word:

all_feature_names = TFIDF.get_feature_names_out()

for word in all_feature_names:
   #let's get the index in the vocabulary
    indx = TFIDF.vocabulary_.get(word)
    
    #get the score
    idf_score = TFIDF.idf_[indx]    
    print(f"{word} : {idf_score}")
    
    
#let's print the transformed output from tf-idf
print(Transformed_output.toarray())


'''Problem Statement: Given a description about a product sold on e-commerce website, classify it in one of the 4 categories
Dataset Credits: https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification

This data consists of two columns.
Text	Label
Indira Designer Women's Art Mysore Silk Saree With Blouse Piece (Star-Red) This Saree Is Of Art Mysore Silk & Comes With Blouse Piece.	Clothing & Accessories
IO Crest SY-PCI40010 PCI RAID Host Controller Card Brings new life to any old desktop PC. Connects up to 4 SATA II high speed SATA hard disk drives. Supports Windows 8 and Server 2012	Electronics
Operating Systems in Depth About the Author Professor Doeppner is an associate professor of computer science at Brown University. His research interests include mobile computing in education, mobile and ubiquitous computing, operating systems and distribution systems, parallel computing, and security.	Books
Text: Description of an item sold on e-commerce website
Label: Category of that item. Total 4 categories: "Electronics", "Household", "Books" and "Clothing & Accessories", which almost cover 80% of any E-commerce website.'''

df = pd.read_csv('C:\\ML\\DataSet\\Ecommerce_data.csv')
print(df[:10])
print(df['label'].value_counts())

df['label_num'] = df['label'].map({"Household":0,
                                   "Electronics":1,
                                   "Clothing & Accessories":2,
                                   "Books":3
    
})

print(df[:10])

X_train,X_test,y_train,y_test = train_test_split(df['Text'],df['label_num'],test_size=0.2,random_state=33)



#1. create a pipeline object
clf = Pipeline([
     ('senthil',TfidfVectorizer()),    
     ('kumar', KNeighborsClassifier())         
])

clf.fit(X_train,y_train)
y_predict = clf.predict(X_test)

print('Classification report using KNeighorsClassifer the performance is:\n',classification_report(y_test,y_predict))

print(X_test[:5])

#2. create a pipeline object
clf = Pipeline([
     ('senthil',TfidfVectorizer()),    
     ('kumar', MultinomialNB())         
])

clf.fit(X_train,y_train)
y_predict = clf.predict(X_test)

print('Classification report using MultinomialNB the performance is:\n',classification_report(y_test,y_predict))


#3. create a pipeline object
clf = Pipeline([
     ('senthil',TfidfVectorizer()),    
     ('kumar', RandomForestClassifier())         
])

clf.fit(X_train,y_train)
y_predict = clf.predict(X_test)

print('Classification report using RandomForestClassifier the performance is:\n',classification_report(y_test,y_predict))

#preprocessing the text 
nlp = spacy.load('en_core_web_sm')
clean_text = []

def preprocess(text):
    # remove stop words and lemmatize the text
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    
    return " ".join(filtered_tokens)

df['preprocessed_text'] = df['Text'].apply(preprocess)  
print(df[:10])

X_train,X_test,y_train,y_test = train_test_split(df['preprocessed_text'],df['label_num'])

clf = Pipeline([('TF-IDF_Vectorizer',TfidfVectorizer()),
                ('NB',MultinomialNB())])

clf.fit(X_train,y_train)
y_predict = clf.predict(X_test)

print('Classification report after preprocessing using MultinomialNb performance are:\n',classification_report(y_test,y_predict))


clf = Pipeline([('TF-IDF_Vectorizer',TfidfVectorizer()),
                ('KN',KNeighborsClassifier())])

clf.fit(X_train,y_train)
y_predict = clf.predict(X_test)

print('Classification report after preprocessing using KNeighborsClassifier performance are:\n',classification_report(y_test,y_predict))


clf = Pipeline([('TF-IDF_Vectorizer',TfidfVectorizer()),
                ('RCF',RandomForestClassifier())])

clf.fit(X_train,y_train)
y_predict = clf.predict(X_test)

print('Classification report after preprocessing using RandomForestClassifier performance are:\n',classification_report(y_test,y_predict))

'''If you compare above classification report with respect to RandomForest Model with the one from unprocessed text, 
you will find some improvement in the model that uses preprocessed cleaned up text. The F1 score improved in the case 
of preprocessed data. Hence we can conclude that for this particular problem using preprocessing (removing stop words, 
lemmatization) is improving the performance of the model.'''

#confusion_matrix
cm = confusion_matrix(y_test,y_predict)

#plot confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Prediction')
plt.ylabel('Truth')