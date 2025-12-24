import pandas as pd 
import torch
from torch import nn, optim
import seaborn as sns
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
data = sns.load_dataset('iris')
print(data['species'].unique())
data['species'] = data['species'].map({'setosa':0,'virginica':1,'versicolor':2})
print(data)

X = data.drop(['species'],axis=1)
y = data['species']

# we need to change to torch data type 

X = torch.tensor(data.drop(['species'],axis=1).values,dtype=torch.float32)
#y = torch.tensor(data['species'].values,dtype=torch.float32).view(X.shape[0],1)
y = torch.tensor(pd.get_dummies(data[['species']],columns=['species'],dtype='int').values,dtype=torch.float32)
print(X)
print(y)

class iris_SimpleNN(nn.Module):
    def __init__(self, input_neurons,output_neurons):
        super(iris_SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_neurons,4)
        self.fc2 = nn.Linear(4,4)
        self.fc3 = nn.Linear(4,output_neurons)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
    def forward(self,X):
        X = self.sigmoid(self.fc1(X))
        X = self.sigmoid(self.fc2(X))
        X = self.softmax(self.fc3(X))
        return X
    
input_neurons = X.shape[1]
output_neurons = len(data['species'].unique())

model = iris_SimpleNN(input_neurons,output_neurons)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.001)
num_epoch = 20000
for epoch in range(1,num_epoch+1):
    optimizer.zero_grad()
    output = model(X)
    loss   = criterion(output,y)
    loss.backward()
    optimizer.step() # for wait update 
    print(f"{epoch} / {num_epoch}, Loss----> {loss.item()}")
    
prediction =  torch.argmax(output,dim=1)
actual = torch.argmax(y,dim=1)
accuracy   = (prediction == actual).sum().item() * 100 / len(y)
print('Accuracy is:',accuracy)

accuracy   = accuracy_score(actual,prediction)
precision  = precision_score(actual,prediction,average='weighted')
recall     = recall_score  (actual,prediction,average='weighted')
f1    = f1_score(actual,prediction,average='weighted')

print('Accuracy is:',accuracy)
#print('precision is:',precision)
print('Recall is:',recall)
print('F1 score is:',f1)
