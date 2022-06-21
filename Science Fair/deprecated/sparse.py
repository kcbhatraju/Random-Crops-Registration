# Mentored by Dr. Nathan Jacobs (https://jacobsn.github.io)
# Images from Skyfinder Dataset (https://cs.valdosta.edu/~rpmihail/skyfinder/)

## USES FOLDERS REMOVED TO REDUCE SIZE OF REPO

import warnings
warnings.filterwarnings("ignore") 

import glob
import itertools
import random
from math import sqrt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

IM_ORIG_SIZE = (125,125)
IM_CROP_SIZE = (100,100)

class Cropping(Dataset):
    def __init__(self, root):
        self.transform = transforms.Compose([transforms.Grayscale(),
                                             transforms.Resize(IM_ORIG_SIZE),
                                             transforms.ToTensor(),
                                             transforms.GaussianBlur(5)
                                            ])
        self.images = [self.transform(Image.open(file)) for file in root] # transform and open images in image list
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        full = self.images[idx]
        where = [RandomCrop.get_params(full, IM_CROP_SIZE)] # (y, x, 256, 256)
        where.append(tuple(pixel+random.randint(1,5) for pixel in where[0][:2])+where[0][2:]) # (y+randint, x+randint, 256, 256)
        
        crops = [TF.crop(full, *place) for place in where]
        shift = sqrt((where[0][0]-where[1][0])**2+(where[0][1]-where[1][1])**2) # sqrt((diff_y)^2 + (diff_x)^2)
        return (torch.cat(crops), torch.tensor([shift])) # (imgs, labels)

trainfull = [glob.glob(f"{dir}*.jpg") for dir in glob.glob("cameras/*/")[2:6]]
files = list(itertools.chain.from_iterable(trainfull))

random.shuffle(files)
train = DataLoader(Cropping(files), batch_size=32)

testfull = [glob.glob(f"{dir}*.jpg") for dir in glob.glob("cameras/*/")[9:11]]
files = list(itertools.chain.from_iterable(testfull))

random.shuffle(files)
test = DataLoader(Cropping(files), batch_size=32)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        torch.set_grad_enabled(True)
        
        self.conv1 = nn.Conv2d(2,20,7,stride=2,padding="valid")
        self.bn1 = nn.BatchNorm2d(20, affine=False)
        self.conv2 = nn.Conv2d(20,21,7,stride=2,padding="valid")
        self.bn2 = nn.BatchNorm2d(21, affine=False)
        self.conv3 = nn.Conv2d(21,11,7,stride=2,padding="valid")
        self.bn3 = nn.BatchNorm2d(11, affine=False)
        self.avgpool = nn.AvgPool2d((3,3))
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(44,1)
    
    def forward(self, x):
        x = self.bn1(F.gelu(self.conv1(x)))
        x = self.bn2(F.gelu(self.conv2(x)))
        x = self.bn3(F.gelu(self.conv3(x)))
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


model = Network()
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 250
trainacc, testacc = [], []
for e in range(epochs):
    print(f"Epoch {e+1}/{epochs}")
    train_acc = 0
    for imgs, labels in train:
        model.train()
        optimizer.zero_grad()
        
        probs = model.forward(imgs)
        
        loss = criterion(probs, torch.tensor(labels))
        loss.backward()
        optimizer.step()
        for idx in range(len(probs)):
            train_acc += abs(probs[idx].item()-labels[idx])
    else:
        train_acc /= len(train)*32
        trainacc.append(train_acc)
        
        test_acc = 0
        for imgs, labels in test:
            model.eval()
            probs = model.forward(imgs)
            for idx in range(len(probs)):
                test_acc += abs((probs[idx].item()-labels[idx]))
        
        test_acc /= len(test)*32
        testacc.append(test_acc)

print(trainacc[-1])
print(testacc[-1])
plt.plot(trainacc, label="Train")
plt.plot(testacc, color="green", label="Test")
plt.legend()
plt.title("Loss of Neural Network Over Time")
plt.xlabel("Epoch")
plt.ylabel("Error Distance (px)")
plt.show()


accuracies = [[0,0] for _ in range(7)]
for images, labels in test:
    model.eval()
    probs = model.forward(images)
    
    loss = [(abs(probs[idx]-labels[idx])/labels[idx]).item() for idx, _ in enumerate(labels)]
    
    for idx, _ in enumerate(loss):
        accuracies[int(round(labels[idx].item()))-1][0] += loss[idx]
        accuracies[int(round(labels[idx].item()))-1][1] += 1

for idx in range(7):
    if(accuracies[idx][1] > 0): accuracies[idx] = 100*(1-accuracies[idx][0]/accuracies[idx][1])
    else: accuracies[idx] = 0

accuracies  = {k+1: v for k, v in enumerate(accuracies)}

plt.barh(list(accuracies.keys()), list(accuracies.values()), color="tab:blue")
plt.title("Accuracy of Neural Network over Distance")
plt.xlabel("Accuracy (%)")
plt.ylabel("Pixel Distance (px)")
plt.show()
