import numpy as np
from PIL import Image


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
            im = Image.open(self.training_set + '%d.jpg' % (self.training_number,))
        except FileNotFoundError:
            raise FileNotFoundError
        img = np.array(im)
        self.data = img
        return img

    def GetValidationData(self):
        self.validation_number += 1
        try:
            im = Image.open(self.validation_set + '%d.jpg' % (self.validation_number,))
        except FileNotFoundError:
            raise FileNotFoundError
        img = np.array(im)
        self.data = img
        return img