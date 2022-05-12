# Mentored by Dr. Nathan Jacobs (https://jacobsn.github.io)
# Images from Skyfinder Dataset (https://cs.valdosta.edu/~rpmihail/skyfinder/)

import warnings

warnings.filterwarnings("ignore")

import glob
from random import randint

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
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, i):
        bigimg = self.imgs[i]
        
        firstcrop = list(RandomCrop.get_params(bigimg, IM_CROP_SIZE))
        if firstcrop[1] > 105: firstcrop[1] = (firstcrop[1])/125*105 # edge case handling [if firstcrop x is > 110 then secondcrop x can go up to > 110+15=125 (off of picture)]
        firstcrop = tuple(firstcrop)
        
        secondcrop = list(firstcrop)
        shift = randint(1,15)
        secondcrop[1] = firstcrop[1] + shift
        secondcrop = tuple(secondcrop)
        
        self.im1 = TF.crop(bigimg, *firstcrop)
        self.im2 = TF.crop(bigimg, *secondcrop)
        
        return (cat([self.im1,self.im2]), tensor(shift)) # (imgs, labels)
    
    def get_imgs(self, i):
        self.__getitem__(i)
        return [self.im1, self.im2]


cam = glob.glob("cameras/10870/*.jpg")
trainroot = cam[:3*len(cam)//4]
testroot = cam[3*len(cam)//4:]

train = DataLoader(Crop(trainroot), batch_size=32, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        set_grad_enabled(True)
        
        self.drop = nn.Dropout(p=0.2)
        self.conv1 = nn.Conv2d(2,20,5,stride=2)
        self.bn1 = nn.BatchNorm2d(20,affine=False)
        self.conv2 = nn.Conv2d(20,19,5,stride=2)
        self.bn2 = nn.BatchNorm2d(19,affine=False)
        self.conv3 = nn.Conv2d(19,11,5,stride=2)
        self.bn3 = nn.BatchNorm2d(11,affine=False)
        self.avgpool = nn.AvgPool2d((3,3))
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(99,50)
        self.dense3 = nn.Linear(50,1)
    
    def forward(self, x):
        x = self.bn1(F.gelu(self.conv1(x)))
        x = self.bn2(F.gelu(self.conv2(x)))
        x = self.bn3(F.gelu(self.conv3(x)))
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.drop(self.dense1(x))
        x = self.dense3(x)
        return x
    
model = Net()
criterion = nn.SmoothL1Loss()
opt = optim.Adam(model.parameters(), lr=0.01)

ep = 1
for e in range(ep):
    print(f"Epoch {e+1}/{ep}")
    for img, lab in train:
        model.train()
        opt.zero_grad()
        probs = model.forward(img)
        loss = criterion(probs.float(), lab.float()) # convert to dtype float for loss function

        loss.backward()
        opt.step()
        
        print(probs)
        print(lab)


# visualization of predictions (20 images)

test = Crop(testroot)

for i in range(20):
    model.eval()
    img, lab = test[i]
    print("True:", lab)
    # print(img[0].shape)
    probs = model.forward(img[None,:])
    print("Predicted:", probs)

    transform = transforms.ToPILImage()
    im1, im2 = [transform(im) for im in test.get_imgs(i)]

    shift = round(probs.item())
    print("Rounded:", shift)

    plt.imshow(im1)
    plt.show()
    plt.imshow(im2)
    plt.show()

    def concat(im1, im2, shift):
        comb = Image.new('RGB', (im1.width+shift, im1.height))
        comb.paste(im1, (0,0))
        comb.paste(im2, (shift,0))
        return comb

    plt.imshow(concat(im1, im2, shift))
    plt.show()