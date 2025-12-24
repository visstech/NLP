import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer
import torch
from torch import nn,optim
import re
from torch.utils.data import DataLoader,Dataset
data = pd.read_csv('C:\\ML\\DataSet\\sentiment_analysis.csv')
print(data)
df = data[data.columns[[1,2]]]
print(df['label'].value_counts())

stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        text = str(text) if not pd.isna(text) else ""
    text = text =  re.sub(r"http\S+|www\S+|https\S+", "", text)    # remove links
    text = re.sub(r'\S+@\S+', '', text)  # remove emails
    text = re.sub(r'<.*?>', '', text) # remove HTML tags
    text = re.sub(f"[^a-zA-Z\s]"," ",text)
    text = re.sub(r"(.)\1{2,}",r"\1\1",text)
    return text  
df['tweet'] = df['tweet'].apply(clean_text)
print(df['tweet'])
Lemmatizer = WordNetLemmatizer()

def nltk_preprocessing(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [Lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)
df['tweet'] = df['tweet'].apply(nltk_preprocessing)
print(df)

Vectorizer = CountVectorizer()
bow_matrix = Vectorizer.fit_transform(df['tweet'])
data_set = pd.DataFrame(bow_matrix.toarray(),columns=Vectorizer.get_feature_names_out())

data_set.index = df['tweet']
print(data_set)
X = data_set
y = df['label']
print(y)
model = LogisticRegression()
model.fit(X,y)
Predicted = model.predict(X)
print('Accuracy score is:',accuracy_score(y,Predicted))

vocab = list(set([token for text in df['tweet'] for token in text.split()]))
print(vocab)
vocab_len = len(vocab)
vocab_to_index = {word: i+1 for i,word in enumerate(vocab)}
print(vocab_to_index)


def text_to_sequence(text):
    max_length = 10
    seq = [vocab_to_index.get(word, 0) for word in text.split()]
    
    if len(seq) < max_length:
        seq += [0] * (max_length - len(seq))
    else:
        seq = seq[:max_length]   # 🔥 THIS LINE FIXES IT
    
    return seq
    
data = df['tweet'].apply(text_to_sequence)
print(data)

X = torch.tensor(data.tolist(),dtype=torch.long)
y = torch.tensor(df['label'].tolist(), dtype=torch.float).unsqueeze(1)

class SentimentDataset(Dataset):
    def __init__(self,X,y):
        self.text = X
        self.label = y
    def __len__(self):
        return len(self.text)
    def __getitem__(self, index):
        return self.text[index],self.label[index]

class RNNArch(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(RNNArch, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        X = self.embedding(X)
        _, hidden = self.rnn(X)
        output = self.fc(hidden[-1])   # ❌ no sigmoid here
        return output
        
dataset = SentimentDataset(X,y)
dataloader = DataLoader(dataset,batch_size=2,shuffle=True)

model = RNNArch(vocab_size=vocab_len,embed_dim=128,hidden_dim=64,output_dim=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(params=model.parameters(),lr=0.0001)
num_epoch = 10
for epoch in range(1,num_epoch+1):
    model.train()
    correct = 0
    total = 0
    for data,label in dataloader:
        optimizer.zero_grad()
        output = model(data)
        loss   = criterion(output,label)
        loss.backward()
        optimizer.step()
        predict = (torch.sigmoid(output) > 0.5).float()
        correct += (label == predict).sum().item()
        total += label.size(0)
    print(f'{epoch}/{num_epoch} Accuracy score{correct/total}')
    
    

