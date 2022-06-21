# Mentored by Dr. Nathan Jacobs (https://jacobsn.github.io)
# Images from Skyfinder Dataset (https://cs.valdosta.edu/~rpmihail/skyfinder/)

## USES FOLDERS REMOVED TO REDUCE SIZE OF REPO

import warnings
warnings.filterwarnings("ignore") 

import os
import random
from math import sqrt
from statistics import median
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop

def optical_flow(*imgs):
    transform = transforms.Compose([transforms.Grayscale(),
                                    transforms.ToTensor(),
                                    transforms.GaussianBlur(5)
                                    ])
    
    imgs = [transform(img)[None] for img in imgs]
    
    horizontal = torch.tensor([[-1., 0., 1.]])[None, None]
    vertical = torch.tensor([[-1.], [0.], [1.]])[None, None]
    time = torch.full((1, 3), 1.)[None, None]
    
    Ix = F.conv2d(imgs[1], horizontal, padding="same")
    Iy = F.conv2d(imgs[1], vertical, padding="same")
    It = (F.conv2d(imgs[1], time, padding="same")-F.conv2d(imgs[0], time, padding="same"))
    
    structure = [Ix*Ix, Ix*Iy, Iy*Iy, Ix*It, Iy*It]
    sum = torch.full((3, 3), 1.)[None, None]
    for idx, _ in enumerate(structure):
        structure[idx] = torch.squeeze(F.conv2d(structure[idx], sum, padding="same")).tolist()
    
    u, v = [], []
    for i, _ in enumerate(structure[0]):
        for j, _ in enumerate(structure[0]):
            a = structure[0][i][j]
            b = c = structure[1][i][j]
            d = structure[2][i][j]
            e = -structure[3][i][j]
            f = -structure[4][i][j]
            
            denom = a*d-c*b
            if denom != 0:
                x = round((e*d-f*b)/denom, 3)
                y = round((a*f-c*e)/denom, 3)
                u.append(x), v.append(y)
    
    return (u, v)

def loss(*crops, x, y):
    true = (crops[0][0]-crops[1][0], crops[0][1]-crops[1][1])
    ground = sqrt(true[0]**2 + true[1]**2)
    
    predict = (median(x), median(y))
    test = sqrt(predict[0]**2 + predict[1]**2)
    
    loss = abs(test-ground)
    percent = 100*loss/ground
    with open("logs/opt_flow_log.txt", "a") as log:
        log.write(f"True shift: {true}. Predicted shift: {predict}.\n")
        log.write(f"Loss: {round(percent, 2)}%\n\n\n")
    
    return (ground, loss, percent)

files = os.listdir("root/train/")
files.extend(os.listdir("root/test/"))
percentages = []
losses = [[0, 0] for _ in range(7)]
accuracies = [[0, 0] for _ in range(7)]
open("logs/opt_flow_log.txt", "w").close()
for idx, file in enumerate(files):
    root = ("root/test/", "root/train/")[2*idx < len(files)]
    img = Image.open(root+file)
    first_crop_params = RandomCrop.get_params(img, [256, 256])
    second_crop_params = tuple(pixel+random.randint(1,5) for pixel in first_crop_params[:2])+first_crop_params[2:]
    
    first_crop = TF.crop(img, *first_crop_params)
    second_crop = TF.crop(img, *second_crop_params)
    
    u, v = optical_flow(first_crop, second_crop)
    val = loss(first_crop_params, second_crop_params, x=u, y=v)
    
    percentages.append(val[2])
    losses[int(round(val[0]))-1][0] += val[1]
    losses[int(round(val[0]))-1][1] += 1
    
    accuracies[int(round(val[0]))-1][0] += 100-percentages[-1]
    accuracies[int(round(val[0]))-1][1] += 1

with open("logs/opt_flow_log.txt", "a") as log:
    log.write(f"Average Loss: {round(sum(percentages)/len(percentages), 2)}%")

for idx in range(7):
    accuracies[idx] = accuracies[idx][0]/accuracies[idx][1]
    losses[idx] = losses[idx][0]/losses[idx][1]

accuracies  = {k+1: v for k, v in enumerate(accuracies)}
losses  = {k+1: v for k, v in enumerate(losses)}

plt.barh(list(accuracies.keys()), list(accuracies.values()), color="tab:blue")
plt.title("Accuracy of Lucas-Kanade over Distance")
plt.xlabel("Accuracy (%)")
plt.ylabel("Pixel Distance (px)")
plt.show()

plt.barh(list(losses.keys()), list(losses.values()), color="tab:orange")
plt.title("Loss of Lucas-Kanade over Distance")
plt.xlabel("Loss (px)")
plt.ylabel("Pixel Distance (px)")
plt.show()
