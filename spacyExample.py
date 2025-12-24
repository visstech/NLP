import spacy
#python -m spacy download en_core_web_sm we need to download this before running this code.
nlp = spacy.load("en_core_web_sm")
doc = nlp('Dr. Senthilkumar is a great person in the world. Mr. Senthil love programming. His daughter Ms. Aruthra is interested in Science.')

for sentence in doc.sents:
    print(sentence)

print('\nWords in the doc are:\n')
for sentence in doc.sents:
    for word in sentence:
        print(word)

for token in doc:
    print('Tokens are:',token)
    
#Token attributes
doc = nlp("Tony gave two $ to Peter.")
token0 = doc[0]
print(token0)
print(token0.is_alpha)
print(token0.like_num)
token2 = doc[2]
print(token2)
print(token2.like_num)
token3 = doc[3]
print('Is currency:',token3.is_currency)
print('Is Digits:',token3.is_digit)

for token in doc:
    print(token,'==>','idex:',token.i,
          'is_alpha:',token.is_alpha,
          'is_digits:',token.is_digit,
          'like_num:',token.like_num,
          'is_currency:',token.is_currency,
          'like_eamil:',token.like_email)
    

print('\n\nreading in text file:\n')
with open('C:\\ML\\DataSet\\NewLearning\\Namelist.txt','r') as f:
    text = f.readlines()

print('Before joing the text is:\n',text)
text = ' '.join(text)
print('\n\n',text)

emails = []
doc = nlp(text)
for token in doc:
    if token.like_email:
        emails.append(token)

print('Email list from the doc are:\n',emails)
#Support in other languages
#Spacy support many language models. Some of them do not support pipelines though! https://spacy.io/usage/models#languages

nlp = spacy.blank("hi")
doc = nlp("भैया जी! 5000 ₹ उधार थे वो वापस देदो")
for token in doc:
    print(token, token.is_currency)
    
text='''
Look for data to help you address the question. Governments are good
sources because data from public research is often freely available. Good
places to start include http://www.data.gov/, and http://www.science.
gov/, and in the United Kingdom, http://data.gov.uk/.
Two of my favorite data sets are the General Social Survey at http://www3.norc.org/gss+website/, 
and the European Social Survey at http://www.europeansocialsurvey.org/.
'''

# TODO: Write code here
# Hint: token has an attribute that can be used to detect a url
doc = nlp(text=text)
url_list =[]
for token in doc:
    if token.like_url:
        url_list.append(token)
print('URL list in the doc are:\n',url_list)

#(2) Extract all money transaction from below sentence along with currency. Output should be,

#two $

#500 €

import spacy.displacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
# Create matcher and define currency pattern
matcher = Matcher(nlp.vocab)

doc = nlp("Tony gave two $ to Peter, Bruce gave 500 € to Steve")

# TODO: Write code here
# Hint: Use token.i for the index of a token and token.is_currency for currency symbol detection



# Add pattern: [number or word] + [currency symbol]
pattern = [
    {"LIKE_NUM": True},  # e.g., 500
    {"IS_CURRENCY": True}  # €, $, etc.
]
pattern2 = [
    {"POS": "NUM"},       # e.g., two
    {"IS_CURRENCY": True}
]

matcher.add("CURRENCY_PATTERN", [pattern, pattern2])

matches = matcher(doc)
for match_id, start, end in matches:
    span = doc[start:end]
    print(span.text)


#Same is done using this
transactions = "Tony gave two $ to Peter, Bruce gave 500 € to Steve ans three $ "
doc = nlp(transactions)
for token in doc:
    if token.like_num and doc[token.i+1].is_currency:
        print(token.text, doc[token.i+1].text)   


print('pipe_names:\n',nlp.pipe_names)

print('pipeline:\n',nlp.pipeline)

#Blank nlp pipeline

nlp = spacy.blank('en') # Only tokenizer will be there for blank pipeline
print('It will created Blank pipe line:\n',nlp.pipe_names)


doc = nlp("Captain america ate 100$ of samosa. Then he said I can do this all day.")

for token in doc:
    print(token)
    
'''Download trained pipeline
To download trained pipeline use a command such as,

python -m spacy download en_core_web_sm

This downloads the small (sm) pipeline for english language'''

nlp = spacy.load('en_core_web_sm')
print('Pipeline Names are:\n',nlp.pipe_names)
print('\n\n Pipelines are:\n',nlp.pipeline)

'''sm in en_core_web_sm means small. There are other models available as well such as medium, large etc.
Check this: https://spacy.io/usage/models#quickstart'''

doc = nlp("Captain america ate 100$ of samosa. Then he said I can do this all day.")

for token in doc:
    print(token," | ",spacy.explain(token.pos_),' | ', token.lemma_)
    
#Named Entity Recognition

doc = nlp("Tesla Inc is going to acquire twitter for $45 billion")

for ent in doc.ents:
    print('Entity in the documents are:\n',ent.text,ent.label_)

from spacy import displacy

print(displacy.render(doc,style='ent'))

#Trained processing pipeline in French we have not downloaded so will get error
#nlp = spacy.load("fr_core_news_sm")

'''You need to install the processing pipeline for french language using this command,

python -m spacy download fr_core_news_sm '''

# alternative way to use it in the code 
from spacy.cli import download

#download("fr_core_news_sm") once downloaded we can commend this line
nlp = spacy.load("fr_core_news_sm")

print(nlp.pipe_names)

doc = nlp("Tesla Inc va racheter Twitter pour $45 milliards de dollars")

for ent in doc.ents:
    print(ent.text,"|",ent.label_,"|",spacy.explain(ent.label_))

for token in doc:
    print(token, " | ", token.pos_, " | ", token.lemma_)

#Adding a component to a blank pipeline
source_nlp = spacy.load('en_core_web_sm')
nlp = spacy.blank('en')
print('Pipeline before adding pipeline in the blank pipeline:\n',nlp.pipeline)
nlp.add_pipe('ner',source=source_nlp)
print('Pipeline after adding pipeline in the blank pipeline:\n',nlp.pipeline)

doc = nlp("Tesla Inc is going to acquire twitter for $45 billion")
for ent in doc.ents:
    print(ent.text,ent.label_)
    

#Stemming in NLTK Stemming is not available in spacy it is only available in NLTK
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
words = ["eating", "eats", "eat", "ate", "adjustable", "rafting", "ability", "meeting"]
print('Stemming words using nltk:\n\n')
for word in words:
    print(word,'=>',stemmer.stem(word))
    
#Lemmatization in Spacy Lemmatization is available in spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp("Mando talked for 3 hours although talking isn't his thing")
doc = nlp("eating eats eat ate adjustable rafting ability meeting better")
print('Lemmatization using spacy :\n\n')
for token in doc:
    print(token,'=>',token.lemma_)
#Customizing lemmatizer

print('Pipe lines are:\n\n',nlp.pipeline)

ar = nlp.get_pipe('attribute_ruler')

ar.add([[{"TEXT":"Bro"}],[{"TEXT":"Brah"}]],{"LEMMA":"Brother"})
 



doc = nlp("Bro, you wanna go? Brah, don't say no! I am exhausted")
print('After customizing pipeline the lemma for bro and Brah are:\n')
for token in doc:
    print(token,'==>',token.lemma_)
    
#Part of speech POS in spacy there are 8 type of part of speech in english 1. Noun. 2. Pronoun 3. verb, 4. adverb, 5. conjunction 6.interjection 7. Preposition 8.Adjective

doc = nlp("Elon flew to mars yesterday. He carried biryani masala with him")
print('POS - Part of speech in spacy:\n')
for token in doc:
    print(token,'==>',token.pos_,'==>',spacy.explain(token.pos_))
    
''' You can check https://v2.spacy.io/api/annotation for the complete list of pos categories in spacy.

https://en.wikipedia.org/wiki/Preposition_and_postposition

https://en.wikipedia.org/wiki/Part_of_speech'''

doc = nlp("Wow! Dr. Strange made 265 million $ on the very first day")
for token in doc:
    print(token,'==>',token.pos_,'==>',spacy.explain(token.pos_))
    
#Tags will explain more about pos example ADV Adverb

doc = nlp("Wow! Dr. Strange made 265 million $ on the very first day")
print('Example for tag in spacy:\n')
for token in doc:
    print(token,'==>',token.pos_,'==>',spacy.explain(token.pos_),'Tags==>',spacy.explain(token.tag_))

#In below sentences Spacy figures out the past vs present tense for quit 

doc = nlp("He quits the job")

print(doc[1].text, "|", doc[1].tag_, "|", spacy.explain(doc[1].tag_))

doc = nlp("he quit the job")

print(doc[1].text, "|", doc[1].tag_, "|", spacy.explain(doc[1].tag_))

#Removing all SPACE, PUNCT and X token from text 

earnings_text="""Microsoft Corp. today announced the following results for the quarter ended December 31, 2021, as compared to the corresponding period of last fiscal year:

·         Revenue was $51.7 billion and increased 20%
·         Operating income was $22.2 billion and increased 24%
·         Net income was $18.8 billion and increased 21%
·         Diluted earnings per share was $2.48 and increased 22%
“Digital technology is the most malleable resource at the world’s disposal to overcome constraints and reimagine everyday work and life,” said Satya Nadella, chairman and chief executive officer of Microsoft. “As tech as a percentage of global GDP continues to increase, we are innovating and investing across diverse and growing markets, with a common underlying technology stack and an operating model that reinforces a common strategy, culture, and sense of purpose.”
“Solid commercial execution, represented by strong bookings growth driven by long-term Azure commitments, increased Microsoft Cloud revenue to $22.1 billion, up 32% year over year” said Amy Hood, executive vice president and chief financial officer of Microsoft."""

doc = nlp(earnings_text)

filtered_tokens = []

for token in doc:
    if token.pos_ not in ["SPACE", "PUNCT", "X"]:
        filtered_tokens.append(token)
print(filtered_tokens)
print('\n Count of POS in the documents are:\n')
count = doc.count_by(spacy.attrs.POS) # Count of POS 
print(count)

print(doc.vocab[96].text)

for k ,v in count.items():
    print(doc.vocab[k].text,'value is:',v)
    
with open('C:\\ML\\DataSet\\news_story.txt') as f:
    text = f.readlines()
    text = ' '.join(text)
doc = nlp(text=text)

print('List of Noun In the story are:\n\n')
LIST_OF_NOUN = []
numeral_tokens = []
for token in doc:
   if token.pos_ in ("NOUN"):
       LIST_OF_NOUN.append(token)
   elif token.pos_ in("NUM"):
        numeral_tokens.append(token)
      
print(LIST_OF_NOUN)
print('Number list:\n\n',numeral_tokens)

NUM_POS = []
print('Extract all numbers (NUM POS type) in a python list:\n\n')
for token in doc:
     NUM_POS.append(token.pos)

print(NUM_POS)

print('Print a count of all POS tags in this story:\n\n')
Count_POS_TAGS =[]
tags ={}
for token in doc:
    count = doc.count_by(spacy.attrs.POS)
for k, v in count.items():
    tags ={doc.vocab[k].text:v}
    Count_POS_TAGS.append(tags)

print(Count_POS_TAGS)

#NLP Tutorial: Named Entity Recognition (NER)
print(' Named Entity Recognition (NER):\n')
doc = nlp("Tesla Inc is going to acquire twitter for $45 billion")

for ent in doc.ents:
    print(ent.text,'==>',ent.label_,spacy.explain(ent.label_))  # Label is like 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY'
print('List down all the entities:\n')
print(nlp.pipe_labels['ner'])

doc = nlp("Michael Bloomberg founded Bloomberg in 1982")
for ent in doc.ents:
    print(ent.text,'==>',ent.label_,'==>',spacy.explain(ent.label_))

doc = nlp("Tesla Inc is going to acquire Twitter Inc for $45 billion")
for ent in doc.ents:
    print(ent.text, " | ", ent.label_, " | ", ent.start_char, "|", ent.end_char)

print('Setting custom entities:\n\n')
from spacy.tokens import Span
s1 = Span(doc, 0, 1, label="ORG") # Here 0 to 1 mean in the document Tesla
s2 = Span(doc, 5, 6, label="ORG") # Here 5 t0 6 mean in the document Twitter

doc.set_ents([s1, s2], default="unmodified") #setting the entity 
for ent in doc.ents:
    print(ent.text, " | ", ent.label_)

print('1. Extract all the Geographical (cities, Countries, states) names from a given text\n')
text = """Kiran want to know the famous foods in each state of India. So, he opened Google and search for this question. Google showed that
in Delhi it is Chaat, in Gujarat it is Dal Dhokli, in Tamilnadu it is Pongal, in Andhrapradesh it is Biryani, in Assam it is Papaya Khar,
in Bihar it is Litti Chowkha and so on for all other states"""

doc = nlp(text)

Geographical =[]
for ent in doc.ents:
    if ent.label_ in('GPE'):
       print(ent.text,'==>',ent.label_ )
       Geographical.append(ent.text)

print('List of Geographical in the doc are:\n',Geographical)

print('Extract all the birth dates of cricketers in the given Text:\n')
text = """Sachin Tendulkar was born on 24 April 1973, Virat Kholi was born on 5 November 1988, Dhoni was born on 7 July 1981
and finally Ricky ponting was born on 19 December 1974."""

doc = nlp(text)
Birth_dates = []
person = []
for ent in doc.ents:
    if ent.label_ in('DATE'):
        Birth_dates.append(ent.text)
    elif ent.label_ in('PERSON'):
        person.append(ent.text)

print(Birth_dates)

#Text Representation - Bag Of Words (BOW) -> Count vectorizer
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from  sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('C:\ML\DataSet\\spam.csv')

print(df[:4])
print(df.Category.value_counts())

df['spam'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)

print(df[:4])
print(df.shape)
# Train test split

X_Train,X_Test,y_train,y_test = train_test_split(df.Message,df.spam,test_size=0.2,random_state=101)

print(X_Train.shape)

#Create bag of words representation using CountVectorizer 
CV = CountVectorizer()
print('X_Train values:\n',X_Train.values)
print('X_Train Keys:\n',X_Train.keys)
X_Train_CV = CV.fit_transform(X_Train.values)
X_Test_CV = CV.transform(X_Test.values)
print(X_Test_CV.toarray()[:2][1])
print(CV.get_feature_names_out()[1771])
print('Vocabulary:\n',CV.vocabulary_)

X_train_np = X_Train_CV.toarray()
print(X_train_np[0])
print(np.where(X_train_np[0]!=0)) # to display only 1 values
print(X_Train[:4])
print(X_Train[:4][1084])

#Train the naive bayes model
model = MultinomialNB()
model.fit(X_Train_CV,y_train)

#Evaluate Performance 
y_pred = model.predict(X_Test_CV)
print('Model Performance are:\n',classification_report(y_test,y_pred))

emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on emailsparking, exclusive offer just for you. Dont miss this reward!'
]

emails_CV = CV.transform(emails)
New_predict = model.predict(emails_CV)
print(New_predict)

#Train the model using sklearn pipeline and reduce number of lines of code
from sklearn.pipeline import Pipeline 

cf_pipe = Pipeline([('vectorizer', CountVectorizer()),
                    ('nb', MultinomialNB())
                    ])

cf_pipe.fit(X_Train,y_train)
y_predict = cf_pipe.predict(X_Test)
print('Y_predicted value:\n',y_predict)
print('Predict using pipeline for new email:\n')
y_predict_with_pipeline = cf_pipe.predict(emails)
print('y Predicted using pipeline is:\n',y_predict_with_pipeline)

'''
In this Exercise, you are going to classify whether a given movie review is positive or negative.
you are going to use Bag of words for pre-processing the text and apply different classification algorithms.
Sklearn CountVectorizer has the inbuilt implementations for Bag of Words.

This data consists of two columns. - review - sentiment
Reviews are the statements given by users after watching the movie.
sentiment feature tells whether the given review is positive or negative.
'''
df = pd.read_csv('C:\\ML\\DataSet\\movies_sentiment_data.csv')
print(df[:5])

df['Category'] = df['sentiment'].apply(lambda x: 1 if x=='positive' else 0 )

print(df[:5])

X_Train,X_Test,y_train,y_test = train_test_split(df.review,df.Category,test_size=0.2,random_state=41)

new_pipe = Pipeline([('vectorizer', CountVectorizer()),
                    ('nb', MultinomialNB())
                    ])
new_pipe.fit(X_Train,y_train)
imdb_move_predict = new_pipe.predict(X_Test)
print('Movie review prediction:\n',imdb_move_predict)
print('Performance metrics using naive_bayes for Move Review are:',classification_report(y_test,imdb_move_predict))

#1. create a pipeline object
clf = Pipeline([
    ('vectorizer', CountVectorizer()),                                                    #initializing the vectorizer
    ('random_forest', (RandomForestClassifier(n_estimators=50, criterion='entropy')))      #using the RandomForest classifier
])

clf.fit(X_Train,y_train)
imdb_move_predict = clf.predict(X_Test)
print('Performance metrics using RandomForestClassifier for Move Review are:\n',classification_report(y_test,imdb_move_predict))

#1. create a pipeline object
clf = Pipeline([
                
     ('vectorizer', CountVectorizer()),   
      ('KNN', (KNeighborsClassifier(n_neighbors=10, metric = 'euclidean')))   #using the KNN classifier with 10 neighbors 
])

clf.fit(X_Train,y_train)
imdb_move_predict = clf.predict(X_Test)
print('Performance metrics using KNeighborsClassifier for Move Review are:\n',classification_report(y_test,imdb_move_predict))


text = '''Can you write some observations of why model like KNN fails to produce good results unlike RandomForest and MultinomialNB?
As Machine learning algorithms does not work on Text data directly, we need to convert them into numeric vector and feed that into models while training.
In this process, we convert text into a very high dimensional numeric vector using the technique of Bag of words.
Model like K-Nearest Neighbours(KNN) doesn't work well with high dimensional data because with large number of dimensions,
it becomes difficult for the algorithm to calculate distance in each dimension. In higher dimensional space, the cost to calculate distance
becomes expensive and hence impacts the performance of model.The easy calculation of probabilities for the words in corpus(Bag of words)
and storing them in contigency table is the major reason for the Multinomial NaiveBayes to be a text classification friendly algorithm.
As Random Forest uses Bootstrapping(Row and column Sampling) with many decision tree and overcomes the high variance 
and overfitting of high dimensional data and also uses feature importance of words for better classifing the categories.
Machine Learning is like trial and error scientific method, where we keep trying all the possible algorithms we have and 
select the one which give good results and satisfy the requirements like latency, interpretability etc.
Refer these resources to get good idea:

https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
https://analyticsindiamag.com/naive-bayes-why-is-it-favoured-for-text-related-tasks/'''

# Stop words tutorial
from spacy.lang.en.stop_words import STOP_WORDS

df = pd.read_json('C:\\ML\\DataSet\\doj_press.json',lines=True)

print(df[0:10])

df = df[df['topics'].str.len()!=0] #creates data frame where topics is not empty
print(df[0:10])

def preprocess(text):
    doc = nlp(text)
    
    no_stop_words = [token.text for token in doc if not token.is_stop]
    return " ".join(no_stop_words)    


df['new_contents'] = df['contents'].apply(preprocess)

print(df[['new_contents','contents']])