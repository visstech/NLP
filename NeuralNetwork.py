import torch
import torchvision
import torchaudio
from torch import nn, optim
print('Version of torch is',torch.__version__)
print('Version of torchvision is',torchvision.__version__)
print('Version of torchaudio is',torchaudio.__version__)

class SimpleNN(nn.Module):
    
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3,2)
        self.fc2 = nn.Linear(2,1)
    def forward(self,X):   
        X = self.fc1(X)
        X = self.fc2(X)
        return X
    
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.001)# model.parameters are waits and bias ,lr high per parameters
X = torch.rand(100,3) # 100 rows and 3 column
y = torch.rand(100,1)
num_epoch = 100
for epoch in range(1,num_epoch+1):
    optimizer.zero_grad() # at first the difference in the error is going to be a zero.
    output = model(X) # the model will do the forward propagation 
    loss = criterion(output,y)
    loss.backward()  #  here it will find out the error in each layer and 
    optimizer.step() #after getting the error will update the waits here.
    print(f"{epoch}/{num_epoch}, Loss --> {loss.item()}")
    
    
X = torch.rand(1000,7)
y = torch.sum(X,axis=1).view(1000,1) + torch.rand(1000,1)
print(y)

class SimpleNN1(nn.Module):
    
    def __init__(self):
        super(SimpleNN1, self).__init__()
        self.fc1 = nn.Linear(7,5) #layer 0 # here First 7 input layers,5 layers in the second and 4 in the third layers,4th layer 3 and 5yh layer 2 and output layer 1
        self.fc2 = nn.Linear(5,4) # layer 1
        self.fc3 = nn.Linear(4,3) # layer 2
        self.fc4 = nn.Linear(3,2) # layer 3
        self.fc5 = nn.Linear(2,1) # layer 4
    def forward(self,X):   
        X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.fc4(X)
        X = self.fc5(X)
        return X

model = SimpleNN1()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.001)# model.parameters are waits and bias ,lr high per parameters
num_epoch = 100
for epoch in range(1,num_epoch+1):
    optimizer.zero_grad() # at first the difference in the error is going to be a zero.
    output = model(X) # the model will do the forward propagation 
    loss = criterion(output,y)
    loss.backward()  #  here it will find out the error in each layer and 
    optimizer.step() #after getting the error will update the waits here.
    print(f"{epoch}/{num_epoch}, Loss --> {loss.item()}")


#Binary classification problem 

X = torch.rand(1000,7)
y = torch.randint(0,2,(1000,1),dtype=torch.float32) #here start with 0 and end with 2 means 0 to 1 and 1000 rows and 1 columns. always 0 or 1 binary classification output value
print(y)

class SimpleNN2(nn.Module):
    
    def __init__(self):
        super(SimpleNN2, self).__init__()
        self.fc1 = nn.Linear(7,5) #layer 0 # here First 7 input layers,5 layers in the second and 4 in the third layers,4th layer 3 and 5yh layer 2 and output layer 1
        self.fc2 = nn.Linear(5,4) # layer 1
        self.fc3 = nn.Linear(4,3) # layer 2
        self.fc4 = nn.Linear(3,2) # layer 3
        self.fc5 = nn.Linear(2,1) # layer 4
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,X):   
        X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.fc4(X)
        X = self.fc5(X)
        X = self.sigmoid(X)
        return X

model = SimpleNN2()
criterion = nn.BCELoss() #Binary cross entropy loss  for binary class classification if it is multi class classification we have to use cross entropy loss function here.
optimizer = optim.SGD(model.parameters(),lr=0.001)# model.parameters are waits and bias ,lr high per parameters
num_epoch = 100
for epoch in range(1,num_epoch+1):
    optimizer.zero_grad() # at first the difference in the error is going to be a zero.
    output = model(X) # the model will do the forward propagation 
    loss = criterion(output,y)
    loss.backward()  #  here it will find out the error in each layer and 
    optimizer.step() #after getting the error will update the waits here.
    print(f"{epoch}/{num_epoch}, Loss --> {loss.item()}")
