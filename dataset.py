import numpy as np
import imageio as reader
import os
import scipy.io as sio
from PIL import Image
from torchvision import transforms

INPUT_SIZE = (448, 448)

class Signature1():
    def __init__(self, root, is_train=True, data_len=None):
        pass
 

    def __getitem__(self, index):
        if self.is_train:
            img, target = None
            # img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            # img = transforms.RandomCrop(INPUT_SIZE)(img)
            # img = transforms.RandomHorizontalFlip()(img)
            img = transforms.Resize((550, 550))(img)
            img = transforms.RandomCrop(INPUT_SIZE, padding=8)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            # img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)

        else:
            img, target = None
            # img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            # img = transforms.CenterCrop(INPUT_SIZE)(img)
            # img = transforms.ToTensor()(img)
            # img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            img = transforms.Resize((550, 550))(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img) 

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)
