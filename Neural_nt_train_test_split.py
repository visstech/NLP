import pandas as pd 
import torch
from torch import nn, optim
import seaborn as sns
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
data = sns.load_dataset('iris')
print(data['species'].unique())
 
print(data)

X = data.drop(['species'],axis=1)
y = data['species']

# we need to change to torch data type 

X = data.drop(['species'],axis=1) 
#y = torch.tensor(data['species'].values,dtype=torch.float32).view(X.shape[0],1)
y = pd.get_dummies(data[['species']],columns=['species'],dtype='int') 
print(X)
print(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=101)
X_train = torch.tensor(X_train.values,dtype=torch.float32)
X_test = torch.tensor(X_test.values,dtype=torch.float32)
y_train = torch.tensor(y_train.values,dtype=torch.float32)
y_test = torch.tensor(y_test.values,dtype=torch.float32)

class iris_SimpleNN(nn.Module):
    def __init__(self):
        super(iris_SimpleNN, self).__init__()
        self.arch = nn.Sequential(
            nn.Linear(4,16),
            nn.ReLU(),
            #nn.Dropout(0.4), #Method to reduce the overfitting 0.4 means 40%
            nn.Linear(16,32),
            nn.ReLU(),
            #nn.Dropout(0.3), # Here in this model 30% dropout it will dropout randomly.
            nn.Linear(32,3),
            nn.Softmax(dim=1)
       )
    def forward(self,X):
         output = self.arch(X)
         return output
    

model = iris_SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.001,weight_decay=0.0001) #weight_decay is to reduce the learning rate parameter values.
num_epoch = 20000
for epoch in range(1,num_epoch+1):
    optimizer.zero_grad()
    output = model(X_train)
    loss   = criterion(output,y_train)
    loss.backward()
    optimizer.step() # for wait update 
    print(f"{epoch} / {num_epoch}, Loss----> {loss.item()}")
    
prediction =  torch.argmax(output,dim=1)
actual = torch.argmax(y_train,dim=1)
accuracy   = (prediction == actual).sum().item() * 100 / len(y_train)
print('Accuracy is:',accuracy)

accuracy   = accuracy_score(actual,prediction)
precision  = precision_score(actual,prediction,average='weighted')
recall     = recall_score  (actual,prediction,average='weighted')
f1    = f1_score(actual,prediction,average='weighted')

print('Accuracy is:',accuracy)
#print('precision is:',precision)
print('Recall is:',recall)
print('F1 score is:',f1)


with torch.no_grad():
    output = model(X_test)
    Loss  = criterion(output,y_test)
print('Loss on test data is:',Loss)