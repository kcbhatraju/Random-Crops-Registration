# Mentored by Dr. Nathan Jacobs (https://jacobsn.github.io)
# Images from Skyfinder Dataset (https://cs.valdosta.edu/~rpmihail/skyfinder/)

import warnings

warnings.filterwarnings("ignore")

from collections import Counter

import copy
import glob
from random import shuffle, randint

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch import nn, optim, cat, tensor, set_grad_enabled
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torchvision import transforms
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt

IM_ORIG_SIZE = (125, 125)
IM_CROP_SIZE = (100, 100)


class Crop(Dataset):
    def __init__(self, root):
        transform = transforms.Compose([transforms.Grayscale(), transforms.Resize(IM_ORIG_SIZE), transforms.ToTensor(), transforms.GaussianBlur(5)])
        self.imgs = [transform(Image.open(img)) for img in root]
        self.vis = []
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, i):
        bigimg = self.imgs[i]
        self.firstcrop = list(RandomCrop.get_params(bigimg, IM_CROP_SIZE))
        if self.firstcrop[1] > 10: self.firstcrop[1] = int((self.firstcrop[1])/25*10) # edge case handling [if firstcrop x is > 110 then secondcrop x can go up to > 110+15=125 (off of picture)]
        self.firstcrop = tuple(self.firstcrop)
        self.secondcrop = list(self.firstcrop)
        shift = randint(0,15)
        
        self.secondcrop[1] = self.firstcrop[1] + shift # shift in the x direction
        self.secondcrop = tuple(self.secondcrop)
        
        self.im1 = TF.crop(bigimg, *self.firstcrop)
        self.im2 = TF.crop(bigimg, *self.secondcrop)
        
        self.vis.append([self.im1, self.im2, shift])
        return (cat([self.im1,self.im2]), tensor(shift)) # (imgs, labels)


cams = glob.glob("cameras/*/")
ims = []
for cam in cams:
    ims += glob.glob(f"{cam}/*.jpg")

trainroot = ims[:3*len(ims)//4]
shuffle(trainroot)
testroot = ims[3*len(ims)//4:]
shuffle(testroot)

train = DataLoader(Crop(trainroot), batch_size=16, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        set_grad_enabled(True)
        
        self.drop = nn.Dropout(p=0.2)
        self.relu = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv2d(2,300,5,stride=2)
        self.bn1 = nn.BatchNorm2d(2,affine=False)
        self.avgpool = nn.AvgPool2d((3,3))
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(76800,16)
    
    def forward(self, x):
        x = self.relu(self.drop(self.conv1(x)))
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

  
model = Net()
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=0.01)

ep = 50
for e in range(ep):
    for img, lab in train:
        print(f"Epoch {e+1}/{ep}")
        model.train()
        opt.zero_grad()
        probs = model.forward(img)
        loss = criterion(probs, lab) # convert to dtype float for loss function
        
        loss.backward()
        opt.step()
        
        print(loss)

# visualization of predictions (20 images)
test = Crop(testroot)

for i in range(20):
    model.eval()
    img, lab = test[i]
    print("True:", lab)
    probs = model.forward(img[None,:])
    top_p, top_class = probs.topk(1, dim=1)
    print(probs)
    print("Predicted:", top_class)
    
    shift = round(top_class.item())
    print("Rounded:", shift)
    
    transform = transforms.ToPILImage()
    im1, im2 = [transform(im) for im in test.vis[i][:2]]
    
    plt.axis("off")
    plt.imshow(im1)
    plt.show()
    plt.imshow(im2)
    plt.show()
    
    def concat(im1, im2, shift):
        comb = Image.new('RGB', (im1.width+shift, im1.height))
        comb.paste(im1, (0,0))
        comb.paste(im2, (shift,0)) # paste second image "shift" pixels to the right of first image
        return comb
    
    plt.imshow(concat(im1, im2, shift))
    plt.show()
