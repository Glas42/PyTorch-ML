import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from model import NeuralNetwork
from dataset import CustomDataset
import cv2
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes
)
from loss import Loss
import os

PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(PATH, 'dataset')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 10000
CLASSES = 3
SPLIT_SIZE = 7
BOUNDINGBOXES = 2
BATCH_SIZE = 4
IMG_SIZE = 448
LEARNING_RATE = 0.0001
NUM_WORKERS = 1
PIN_MEMORY = False

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes

transform = Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

def main():
    model = NeuralNetwork(split_size=7, boundingboxes=2, classes=CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = Loss(split_size=SPLIT_SIZE, boundingboxes=BOUNDINGBOXES, classes=CLASSES)

    train_dataset = CustomDataset(data_dir=DATA_PATH, split_size=SPLIT_SIZE, boundingboxes=BOUNDINGBOXES, classes=CLASSES, transform=transform)

    test_dataset = CustomDataset(data_dir=DATA_PATH, split_size=SPLIT_SIZE, boundingboxes=BOUNDINGBOXES, classes=CLASSES, transform=transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(EPOCHS):
        model.train()

        mean_loss = []
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = loss_fn(out, y)
            mean_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()       

        model.eval()

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(DEVICE)
                for idx in range(x.shape[0]):
                    bboxes = cellboxes_to_boxes(model(x), split_size=SPLIT_SIZE, classes=CLASSES, device=DEVICE)
                    bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4)
                    image = x[idx].permute(1,2,0).to(DEVICE)
                    image = (image * 255).byte().detach().cpu().numpy()
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    for box in bboxes:
                        box = box[2:]
                        upper_left_x = (box[0] - box[2] / 2) * image.shape[1]
                        upper_left_y = (box[1] - box[3] / 2) * image.shape[0]
                        width = box[2] * image.shape[1]
                        height = box[3] * image.shape[0]
                        cv2.rectangle(image, (int(upper_left_x), int(upper_left_y)), (int(upper_left_x + width), int(upper_left_y + height)), (0, 0, 255), 1)
                    cv2.imshow('image', image)
                    cv2.waitKey(1)

        pred_boxes, target_boxes = get_bboxes(test_loader, model, iou_threshold=0.5, threshold=0.4, split_size=SPLIT_SIZE, classes=CLASSES, device=DEVICE)
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, classes=CLASSES)
    
        print(f"\nTrain mAP: {mean_avg_prec}")
        print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

if __name__ == "__main__":
    main()