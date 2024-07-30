import torchvision.transforms.v2 as transforms
from matplotlib import pyplot as plt
from torchvision import tv_tensors
import albumentations as A
from PIL import Image
import numpy as np
import torchvision
import random
import torch
import os

IMG_SIZE = 140

transform = A.Compose([
    A.RandomCrop(width=round(random.uniform(0.5, 1) * IMG_SIZE), height=round(random.uniform(0.5, 1) * IMG_SIZE)),
    A.Rotate(limit=45, p=0.5),
    A.Resize(IMG_SIZE, IMG_SIZE),
], bbox_params=A.BboxParams(format='yolo'))

PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_PATH = PATH + "\\ModelFiles\\EditedTrainingData"

image = Image.open(DATA_PATH + "\\8.png")
boxes = []
with open(DATA_PATH + "\\8.txt") as f:
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

image = np.array(image)

temp = []
for box in boxes:
    temp.append([box[1], box[2], box[3], box[4], box[0]])
boxes = temp

while True:
    transformed = transform(image=image, bboxes=boxes)
    image1 = transformed["image"]
    boxes1 = transformed["bboxes"]

    plt.figure(figsize=(3, 3))
    img = np.array(image1)
    for box in boxes1:
        x, y, width, height, class_label = box
        x1 = (x - width / 2) * IMG_SIZE
        y1 = (y - height / 2) * IMG_SIZE
        x2 = (x + width / 2) * IMG_SIZE
        y2 = (y + height / 2) * IMG_SIZE
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2))
    plt.imshow(img)
    plt.title("Image with Bounding Boxes")
    plt.show()