# Mentored by Dr. Nathan Jacobs (https://jacobsn.github.io)
# Images from Skyfinder Dataset (https://cs.valdosta.edu/~rpmihail/skyfinder/)

import warnings
warnings.filterwarnings("ignore")

import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch import nn, optim, set_grad_enabled, cat, load, save, tensor
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import RandomCrop
from torchvision.utils import make_grid

IM_ORIG_SIZE = (125, 125)
IM_CROP_SIZE = (100, 100)


class Crop(Dataset):
    def __init__(self, root):
        transform = transforms.Compose([transforms.Grayscale(), transforms.Resize(IM_ORIG_SIZE), transforms.ToTensor(), transforms.GaussianBlur(5)])
        self.imgs = [transform(Image.open(path)) for path in root]
        self.cropped = []
        self.dir = []
        self.paired = []
        for i in range(0,len(self.imgs),2):
            self.paired.append([self.imgs[i],self.imgs[i+1]])
        
        self.present = [transforms.Resize(IM_ORIG_SIZE)(Image.open(path)) for path in root]
        self.bin = []
        for i in range(0,len(self.present),2):
            self.bin.append([self.present[i],self.present[i+1]])
            
        every = list(zip(self.paired,self.bin))
        random.shuffle(every)
        self.paired, self.bin = zip(*every)
    
    def __len__(self):
        return len(self.paired)
    
    def __getitem__(self, i):
        img1, img2 = self.paired[i]
        cimg1, cimg2 = self.bin[i]
        
        dir = random.randint(0,1)
        self.dir.append(dir)
        firstcrop = list(RandomCrop.get_params(img1, IM_CROP_SIZE))
        if firstcrop[dir] > 10: firstcrop[dir] = int((firstcrop[dir])/25*10) # edge case handling [if firstcrop x is > 110 then secondcrop x can go up to > 110+15=125 (off of picture)]
        firstcrop = tuple(firstcrop)
        
        secondcrop = list(firstcrop)
        shift = random.randint(0,15)
        secondcrop[dir] = firstcrop[dir] + shift
        secondcrop = tuple(secondcrop)
        
        im1 = TF.crop(img1, *firstcrop)
        im2 = TF.crop(img2, *secondcrop)
        
        color1 = TF.crop(cimg1, *firstcrop)
        color2 = TF.crop(cimg2, *secondcrop)
        self.cropped.append([color1, color2])
        
        return (cat([im1,im2]), tensor(shift)) # (imgs, labels)


cams = glob.glob("cameras/registered/*/")
trainroot = []
for cam in cams:
    one = glob.glob(f"{cam}/*.jpg")
    random.shuffle(one)
    trainroot += one

shuffims = []
for i in range(0,len(trainroot),2):
    shuffims.append([trainroot[i],trainroot[i+1]])

random.shuffle(shuffims)
trainroot = []
for tup in shuffims:
    [trainroot.append(val) for val in tup]

cams = glob.glob("cameras/testing/*/")
testroot = []
for cam in cams:
    one = glob.glob(f"{cam}/*.jpg")
    random.shuffle(one)
    testroot += one

shuffims = []
for i in range(0,len(testroot),2):
    shuffims.append([testroot[i],testroot[i+1]])

random.shuffle(shuffims)
testroot = []
for tup in shuffims:
    [testroot.append(val) for val in tup]

trainset = Crop(trainroot)
trainloader = DataLoader(trainset, batch_size=16, shuffle=False)
testset = Crop(testroot)


class DispNet(nn.Module):
    def __init__(self):
        super().__init__()
        set_grad_enabled(True)
        
        self.drop = nn.Dropout(p=0.2)
        self.relu = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv2d(2,300,5,stride=2)
        self.conv2 = nn.Conv2d(300,400,5,stride=2)
        self.conv3 = nn.Conv2d(400,500,5,stride=2)
        self.avgpool = nn.AvgPool2d((3,3))
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(4500,16)
    
    def forward(self, x):
        x = self.relu(self.drop(self.conv1(x)))
        x = self.relu(self.drop(self.conv2(x)))
        x = self.relu(self.drop(self.conv3(x)))
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def train(epochs=75):
    model = DispNet()
    try: model.load_state_dict(load("model/checkpoint.txt"))
    except FileNotFoundError: pass
    
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=3e-4)
    
    for e in range(epochs):
        with open("logs/train_log.txt", "a") as log:
            log.write(f"Epoch {e+1}/{epochs} Losses\n")
        for img, lab in trainloader:
            model.train()
            opt.zero_grad()
            probs = model.forward(img)
            loss = criterion(probs, lab)
            with open("logs/train_log.txt", "a") as log:
                log.write(f"{round(loss.item(),2)}\n")
            loss.backward()
            opt.step()
    
    save(model.state_dict(),"model/checkpoint.txt")

def evaluate(type, times=25):
    match type.lower():
        case "train": set, name = trainset, "R"
        case "test": set, name = testset, "T"
        case _: raise ValueError("Invalid evaluation type.")
    
    model = DispNet()
    model.load_state_dict(load("model/checkpoint.txt"))
    summed = 0
    for i in range(min(times,len(set))):
        model.eval()
        img, lab = set[i]
        probs = model.forward(img[None,:])
        _, top_class = probs.topk(1, dim=1)
        
        transform = transforms.ToTensor()
        full1, full2 = [transform(im) for im in set.bin[i]]
        pil1, pil2 = [im.resize((125,125))  for im in set.cropped[i]]
        crop1, crop2 = transform(pil1), transform(pil2)
        
        size = [pil1.width, pil1.height]
        size[1-set.dir[i]] += top_class.item()
        comb = Image.new("RGB", tuple(size))
        
        shift = [0,0]
        shift[1-set.dir[i]] += top_class.item()
        comb.paste(pil1, (0,0))
        comb.paste(pil2, tuple(shift))
        
        imgs = cat((full1.unsqueeze(0),full2.unsqueeze(0),crop1.unsqueeze(0),crop2.unsqueeze(0)))
        imgs = np.transpose(make_grid(imgs,padding=True,normalize=True),(1,2,0))
        
        plt.imshow(imgs)
        plt.show()
        
        match type.lower():
            case "train": orig = "(R)"
            case "test": orig = f"({input('Original: ').upper()})"
        
        match orig:
            case "(R)": state = (abs(top_class.item()-lab.item()) <= 1)
            case "(U)": state = (abs(top_class.item()-lab.item()) > 1)
            case _: raise ValueError("Invalid registration type.")
        
        plt.imshow(comb)
        plt.show()
        
        summed += state
        results = glob.glob("results/*/")
        size = len(glob.glob(f"{results[2]}/*/"))
        file = 2-(state)*((orig == "(R)")+1)
        size = len(glob.glob(f"{results[file]}/*/"))
        path = f"{results[file]}{orig[1]}{size+1}/"
        
        os.mkdir(path)
        imgs = transforms.ToPILImage()(np.transpose(imgs,(2,0,1)))
        imgs.save(f"{path}input{size+1}.png")
        comb.save(f"{path}output{size+1}.png")
        
        with open("logs/evaluate_log.txt", "a") as log:
            log.write(f"{name}{orig[1]}{size+1}. True: {lab.item()}, Predicted: {top_class.item()} {('(Failed)','(Passed)')[state]}\n")
    
    with open("logs/evaluate_log.txt", "a") as log:
        log.write(f"{name} Accuracy: {round(summed/times*100,2)}%\n")
    
if __name__ == "__main__": train()
