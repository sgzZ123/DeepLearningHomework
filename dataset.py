import numpy as np
from PIL import Image
import torch


class Data:
    def __init__(self, FileName='/input_dir/dataset'):
        self.FileName = FileName
        self.training_set = self.FileName + 'training'
        self.validation_set = self.FileName + 'validation'
        self.data = None
        self.training_number = 0
        self.validation_number = 0

    def GetTrainingData(self):
        self.training_number += 1
        try:
            im1 = Image.open(self.training_set + 'input' + '%d.jpg' % (self.training_number,))
            im2 = Image.open(self.training_set + 'label' + '%d.jpg' % (self.training_number,))
        except FileNotFoundError:
            raise FileNotFoundError
        img1 = np.array(im1)
        img1 = torch.from_numpy(img1)
        img2 = np.array(im2)
        img2 = torch.from_numpy(img2)
        self.data = img1
        return img1, img2

    def GetValidationData(self):
        self.validation_number += 1
        try:
            im1 = Image.open(self.validation_set + 'input' + '%d.jpg' % (self.training_number,))
            im2 = Image.open(self.validation_set + 'label' + '%d.jpg' % (self.training_number,))
        except FileNotFoundError:
            raise FileNotFoundError
        img1 = np.array(im1)
        img1 = torch.from_numpy(img1)
        img2 = np.array(im2)
        img2 = torch.from_numpy(img2)
        self.data = img1
        return img1, img2
