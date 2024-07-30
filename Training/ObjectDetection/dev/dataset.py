"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
from PIL import Image
from torchvision import transforms

PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_PATH = PATH + "\\ModelFiles\\EditedTrainingData"

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, files=[], transform=None, SplitSize=7, BoundingBoxes=2, Classes=10):
        self.files = files
        self.transform = transform
        self.SplitSize = SplitSize
        self.BoundingBoxes = BoundingBoxes
        self.Classes = Classes

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_name = self.files[index]
        img_path = os.path.join(DATA_PATH, img_name)
        label_path = os.path.join(DATA_PATH, img_name.replace(img_name.split('.')[-1], 'txt'))
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [float(x) if float(x) != int(float(x)) else int(x) for x in label.replace("\n", "").split()]
                x1 = x - width / 2
                y1 = y - height / 2
                x2 = x + width / 2
                y2 = y + height / 2
                x1 = max(0, min(1, x1))
                y1 = max(0, min(1, y1))
                x2 = max(0, min(1, x2))
                y2 = max(0, min(1, y2))
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                boxes.append([class_label, x, y, width, height])
        image = Image.open(img_path).convert("RGB")
        boxes = torch.tensor(boxes)
        if self.transform != None:
            image, boxes = self.transform(image, boxes)
        else:
            image = transforms.ToTensor()(image)
        label_matrix = torch.zeros((self.SplitSize, self.SplitSize, self.Classes + self.BoundingBoxes * 5))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            i, j = int(self.SplitSize * y), int(self.SplitSize * x)
            if i >= self.SplitSize or j >= self.SplitSize: continue
            x_cell, y_cell = self.SplitSize * x - j, self.SplitSize * y - i
            width_cell, height_cell = (width * self.SplitSize, height * self.SplitSize)
            if label_matrix[i, j, self.Classes] == 0:
                label_matrix[i, j, self.Classes] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, self.Classes + 1:self.Classes + 5] = box_coordinates
                label_matrix[i, j, class_label] = 1
        return image, label_matrix