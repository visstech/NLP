import torch
import torchvision
import torchaudio
from torch import nn, optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ["age","workclass","fnlwgt","education","education_num",
            "marital_status","occupation","relationship","race","sex",
            "capital_gain","capital_loss","hours_per_week","native_country","income"]

adult_data = pd.read_csv(url,header=None,names=columns,skipinitialspace=True)
adult_data.drop(['fnlwgt','education','capital_gain','capital_loss'],axis=1,inplace=True)
adult_data.replace('?',pd.NA,inplace=True)
adult_data.dropna(inplace=True)
print(adult_data)

cat_columns = ['workclass','marital_status','occupation','relationship','race','native_country','income','sex']

encoder = LabelEncoder()
for col in cat_columns:
    adult_data[col] = encoder.fit_transform(adult_data[col])

print('After label encoder:\n',adult_data)
X = torch.tensor(adult_data.drop(['income'],axis=1).values,dtype=torch.float32)
y = torch.tensor(adult_data['income'].values,dtype=torch.float32).view(X.shape[0],1) # here view is to make a list of list each values are changed as list
print(y)

 
 
class SimpleNN1(nn.Module):
    
    def __init__(self,input_neurons,output_neurons):
        super(SimpleNN1, self).__init__()
        self.fc1 = nn.Linear(input_neurons,5) #layer 0 # here First 7 input layers,5 layers in the second and 4 in the third layers,4th layer 3 and 5yh layer 2 and output layer 1
        self.fc2 = nn.Linear(5,4) # layer 1
        self.fc3 = nn.Linear(4,3) # layer 2
        self.fc4 = nn.Linear(3,2) # layer 3
        self.fc5 = nn.Linear(2,output_neurons) # layer 4
    def forward(self,X):   
        X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.fc4(X)
        X = self.fc5(X)
        return X
input_neurons = X.shape[1] # Number of columns available in the X
output_neurons = 1 
model = SimpleNN1(input_neurons,output_neurons)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.001)# model.parameters are waits and bias ,lr high per parameters
num_epoch = 10000
for epoch in range(1,num_epoch+1):
    optimizer.zero_grad() # at first the difference in the error is going to be a zero.
    output = model(X) # the model will do the forward propagation 
    loss = criterion(output,y)
    loss.backward()  #  here it will find out the error in each layer and 
    optimizer.step() #after getting the error will update the waits here.
    print(f"{epoch}/{num_epoch}, Loss --> {loss.item()}")
