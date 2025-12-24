#pip install pillow
from PIL import Image
import torch
import numpy
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms # to transform image data to tensor.
img = Image.open('C://ML//DataSet//NewLearning//NLP//download.jpg')
print(img.size)
transformation = transforms.ToTensor() 
image_data = transformation(img)
print('image shape is:',image_data.shape)

print(image_data)
#back to image
image = torch.tensor(image_data,dtype=torch.float32)
img = (image.permute(1,2,0)).numpy()
plt.imshow(img)
plt.show()
#data = torchvision.datasets.Caltech101(root='/data',transform=torchvision.transforms.ToTensor() ,download=True)

from torchvision.datasets import Caltech101
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch

# Image transforms: Resize, ToTensor, Normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

import torchvision.datasets as datasets

# Specify the root directory where you want to save the dataset
root_dir = './data' 

# Download the dataset
caltech101_dataset = datasets.Caltech101(root=root_dir, download=True)