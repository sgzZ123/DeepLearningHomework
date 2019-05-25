import numpy as np
from PIL import Image
import torch
import os
from torchvision import datasets, models, transforms
import copy


class Data:
    def __init__(self, FileName="D:\\image\\"):
        self.FileName = FileName
        self.training_set = self.FileName + 'training\\'
        self.validation_set = self.FileName + 'validation\\'
        self.data = None
        self.training_number = 0
        self.validation_number = 0



    def GetTrainingData(self):
        self.training_number += 1
        try:
            im1 = Image.open(self.training_set + 'input\\' + '%d.jpg' % (self.training_number,))
            im2 = Image.open(self.training_set + 'label\\' + '%d.jpg' % (self.training_number,))
        except FileNotFoundError:
            raise FileNotFoundError
        img1 = np.array(im1)
        img1 = torch.from_numpy(img1)
        img1 = img1.unsqueeze(0)
        img1 = img1.unsqueeze(0)
        img2 = self.convert(im2)
        self.data = img1
        return img1, img2

    def GetValidationData(self):
        self.validation_number += 1
        try:
            im1 = Image.open(self.validation_set + 'input\\' + '%d.jpg' % (self.validation_number,))
            im2 = Image.open(self.validation_set + 'label\\' + '%d.jpg' % (self.validation_number,))
        except FileNotFoundError:
            raise FileNotFoundError
        img1 = np.array(im1)
        img1 = torch.from_numpy(img1)
        img1 = img1.unsqueeze(0)
        img1 = img1.unsqueeze(0)
        img2 = self.convert(im2)
        self.data = img1
        return img1, img2

    def convert(self, im):
        a_ = np.zeros((224, 224), dtype=np.int)
        b_ = np.zeros((224, 224), dtype=np.int)
        pix = im.load()
        for x in range(224):
            for y in range(224):
                r, g, b = pix[x,y]
                R = gamma(r/255.0)
                G = gamma(g/255.0)
                B = gamma(b/255.0)
                X = 0.412453 * R + 0.357580 * G + 0.180423 * B
                Y = 0.212671 * R + 0.715160 * G + 0.072169 * B
                Z = 0.019334 * R + 0.119193 * G + 0.950227 * B
                X /= 0.95047
                Y /= 1.0
                Z /= 1.08883
                FX = f(X)
                FY = f(Y)
                FZ = f(Z)
                a_[x,y] = 500 * (FX - FY)
                b_[x,y] = 200 * (FY - FZ)

        a_ = torch.from_numpy(a_)
        a_ = a_.unsqueeze(0)
        a_ = a_.unsqueeze(0)
        b_ = torch.from_numpy(b_)
        b_ = b_.unsqueeze(0)
        b_ = b_.unsqueeze(0)
        im = torch.cat((a_,b_), 1)
        return im


def gamma(im_channel):
    return ((im_channel+0.055)/1.055)**2.4 if im_channel > 0.04045 else im_channel / 12.92


def f(im_channel):
    return im_channel ** 1 / 3 if im_channel > 0.008856 else 7.787 * im_channel + 0.137931
