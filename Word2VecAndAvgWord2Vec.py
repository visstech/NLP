import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from gensim.models import Word2Vec
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Load dataset
message = pd.read_csv('C:\\ML\\DataSet\\SMSSpamCollection.txt', 
                      sep='\t', names=['label', 'messages'])

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

corpus = []

# Text Preprocessing
for msg in message['messages']:
    review = re.sub('[^a-zA-Z]', ' ', msg)
    review = review.lower().split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in stop_words]
    corpus.append(review)

# ➤ Train Word2Vec model
w2v_model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1)

word_vectors = w2v_model.wv

# ➤ Average Word2Vec representation
def avg_word2vec(words):
    vectors = [word_vectors[word] for word in words if word in word_vectors]
    return np.mean(vectors, axis=0) if len(vectors) > 0 else np.zeros(100)

X = np.array([avg_word2vec(words) for words in corpus])

y = message['label'].map({'ham': 0, 'spam': 1})

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----------------- Logistic Regression -----------------
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

print("\n📌 Logistic Regression with Word2Vec")
print("Accuracy:", accuracy_score(y_test, log_pred))
print(confusion_matrix(y_test, log_pred))
print(classification_report(y_test, log_pred))

# ----------------- XGBoost -----------------
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

print("\n📌 XGBoost with Word2Vec")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print(confusion_matrix(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))

# ----------------- Random Forest -----------------
rf_model = RandomForestClassifier(n_estimators=400)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\n📌 Random Forest with Word2Vec")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
