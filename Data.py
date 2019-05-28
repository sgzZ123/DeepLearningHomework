import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from skimage.color import lab2rgb, rgb2lab, rgb2gray


class GrayscaleImageFolder(datasets.ImageFolder):
    def __getitem__(self, item):
        path, target = self.imgs[item]
        img = self.loader(path)
        if self.transform is not None:
            img_original = self.transform(img)
            img_original = np.asarray(img_original)
            img_lab = rgb2lab(img_original)
            img_lab = (img_lab + 128) / 255
            img_ab = img_lab[:,:,1:3]
            img_ab = torch.from_numpy(img_ab.transpose((2,0,1))).float()
            img_original = rgb2gray(img_original)
            img_original = torch.from_numpy(img_original).unsqueeze(0).float()
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img_original, img_ab, target


def to_rgb(grayscale_input, ab_input,save_path=None,save_name=None):
    color_image = torch.cat((grayscale_input, ab_input), 1).squeeze(0).numpy()
    color_image = color_image.transpose((1,2,0))
    color_image[:,:,0:1] = color_image[:,:,0:1] * 100
    color_image[:,:,1:3] = color_image[:,:,1:3] *255 -128
    color_image =lab2rgb(color_image.astype(np.float64))
    grayscale_input = grayscale_input.squeeze().numpy()
    if save_path is not None and save_name is not None:
        plt.imsave(arr=grayscale_input,
                   fname='{}{}'.format(save_path['grayscale'],save_name),cmap='gray')
        plt.imsave(arr=color_image,fname='{}{}'.format(save_path['colorized'],save_name))



train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip()
])
train_imagefolder = GrayscaleImageFolder('D:\\image\\training',train_transforms)
train_loader = torch.utils.data.DataLoader(train_imagefolder, batch_size=1,shuffle=True)

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224)
])
val_imagefolder = GrayscaleImageFolder('D:\\image\\validation',val_transforms)
val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=1,shuffle=True)

dataloader = {'train':train_loader, 'valid':val_loader}
dataset_sizes = {'train':len(train_imagefolder),'valid':len(val_imagefolder)}