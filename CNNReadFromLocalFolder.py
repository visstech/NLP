
from torchvision import datasets,transforms
import matplotlib.pyplot as plt 
from torch import nn,optim
from torch.utils.data import DataLoader

data = datasets.ImageFolder(root='C:/ML/CNN/train_cancer',transform=transforms.ToTensor())
print(data)
img,Label = data[0]  #when ever the image in the tensor it is represented as channel, hieght,width
img=img.permute(1,2,0) # here we need hieght,width and channel so give 1 for hieght,2 for width and 0 for channe
plt.imshow(img)
plt.show()

from PIL import Image
import torch
from torchvision import transforms 

img = Image.open('C:/ML/DataSet/NewLearning/NLP/Lotus.jpg')
transformations = transforms.ToTensor()
tensor_imag = transformations(img)
img = torch.tensor(tensor_imag,dtype=torch.float32)
img = (img.permute(1,2,0)).numpy()
plt.imshow(img)
plt.show()

# only one channel gray color image
img = Image.open('C:/ML/DataSet/NewLearning/NLP/Lotus.jpg')
transformations = transforms.Compose([transforms.Grayscale(num_output_channels=1)
                                      ,transforms.ToTensor()])
tensor_imag = transformations(img)
img = torch.tensor(tensor_imag,dtype=torch.float32)
img = (img.permute(1,2,0)).numpy()
plt.imshow(img,cmap='gray')
plt.show()
print(img.shape)# (407, 612, 1) here 1 is the number of channel

#Apply filter on the image
img = Image.open('C:/ML/DataSet/NewLearning/NLP/Lotus.jpg')
transformations = transforms.Compose([transforms.Grayscale(num_output_channels=1)
                                      ,transforms.ToTensor()])
#above we are transform the image as block and white and also converting to tensor values.

tensor_imag = transformations(img)
print(tensor_imag.shape) # torch.Size([1, 407, 612]) here 1 channel 407 hight, 612 width
# To apply filter on the image we need batchsize,channel,hight,width
tensor_imag = tensor_imag.unsqueeze(0) # now it will create 4 
print(tensor_imag.shape) #torch.Size([1, 1, 407, 612]) here 1 batch,1 channel 407 hight, 612 width 
# we need to include batch size to apply filter so we do unsqueeze in the tensor_image so that can apply filter.
#Note squeeze means removing one dimension and unsqueeze means adding one more dimension
sobel = torch.tensor([[-1,0,1],
                      [-2,0,2],
                      [-1,0,1]
                      ],dtype=torch.float32).unsqueeze(0).unsqueeze(0)# Here 0 means at the index number 0 to add batch size
                     # here [[-1,0,1],[-2,0,2],[-1,0,1]]
                     # two dimension [width,hight] first unsqueeze(0) will create [channel,width,hight]
                     # next unsqueeze(0) will create [batchsize,channel,width,hight] which is ready to apply filter 
output = torch.nn.functional.conv2d(tensor_imag,sobel,stride=1) # after applying filter or convalution the output
print(output.shape) #after applying filter image size has reduced now.
# stride=1 means how many steps we move our filter 2 two steps at a time like that.

# To create show the image in matplotlib we need [H,W,C]
output = output.squeeze(0) # this will remove batch size from the output now [C,H,W]
print('After squeeze output now:',output.shape)
output = output.permute(1,2,0) #it will change from [C,H,W] to [H,W,C] which we can use to plot
plt.imshow(output,cmap='gray')
plt.show()

#to resize all the images as same size below code is used.
transformation = transforms.Compose([transforms.Resize((224,224)),#A tuple of two integers (height, width) → the image will be resized to exactly that size
                                      transforms.ToTensor()])
dataset = datasets.ImageFolder(root='C:/ML/CNN/train_cancer',transform=transformation)
train_data = DataLoader(dataset,batch_size=64) # create small small batches with size 64
img,Label = dataset[0]
print(img.shape) #(3 channels for RGB, 224 height, 224 width)

class CNN_Cancer(nn.Module):

    def __init__(self, ):
        super(CNN_Cancer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3)
        # width = (width_input+2+padding-kernel_size//stride + 1) ==> 222*222 actual size was 224 * 224
        #hight  = (hight_input+2+padding-kernel_size//stride + 1) ==> 222*222 actual size was 224 * 224
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3) # 220 * 220
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3) # 218 * 218 
        #fully connected neural network
        self.fc1 = nn.Linear(218*218*64,514) # 514 hidden layers
        self.fc2 = nn.Linear(514,64)
        self.fc3 = nn.Linear(64,2)
        self.dropout = nn.Dropout(0.3)
        self.relu  = nn.ReLU()
        self.softmax = nn.Softmax()
        
        
    def forward(self,X):
        X =self.conv1(X)
        X =self.conv2(X)
        X =self.conv3(X)
        X = X.view(-1,218*218*64) #Flatten means 2d to 1d so that we can pass to neural network
        #Here -1 represent to each row
        X = self.relu(self.fc1(X))
        X = self.dropout(X)
        X = self.relu(self.fc2(X))
        X = self.relu(self.fc3(X))
        x = self.softmax(X,dim=X)
        return X    
        
model = CNN_Cancer()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.001,weight_decay=0.001)
num_epoch= 10
for epoch in range(1,num_epoch+1):
    optimizer.zero_grad()
    total_error = 0
    for imgs,Labels in train_data:
        outputs =model(imgs)
        losses = criterion(outputs,Labels)
        losses.backward()
        optimizer.step()
        total_error  = total_error + losses.item()
    print(f" {epoch} / {num_epoch} ,Loss==> {total_error}")
