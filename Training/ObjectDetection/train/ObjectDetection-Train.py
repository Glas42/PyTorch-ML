import datetime
print(f"\n----------------------------------------------\n\n\033[90m[{datetime.datetime.now().strftime('%H:%M:%S')}] \033[0mImporting libraries...")

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from collections import Counter
import torch.optim as optim
import multiprocessing
import torch.nn as nn
from PIL import Image
import numpy as np
import threading
import random
import shutil
import torch
import time
import cv2

# Constants
PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_PATH = PATH + "\\ModelFiles\\EditedTrainingData"
MODEL_PATH = PATH + "\\ModelFiles\\Models"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 300
BATCH_SIZE = 4
CLASSES = 3
SPLIT_SIZE = 7
BOUNDINGBOXES = 2
IMG_SIZE = 448
IMG_CHANNELS = ['Grayscale', 'Binarize', 'RGB', 'RG', 'GB', 'RB', 'R', 'G', 'B'][2]
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5
NMS_ACROSS_ALL_CLASSES = True
LEARNING_RATE = 0.0001
MAX_LEARNING_RATE = 0.001
TRAIN_VAL_RATIO = 0.8
NUM_WORKERS = 0
PATIENCE = 100
SHUFFLE = True
PIN_MEMORY = True
DROP_LAST = True

ARCHITECTURE = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

IMG_COUNT = 0
for file in os.listdir(DATA_PATH):
    if file.endswith(".jpg"):
        IMG_COUNT += 1
if IMG_COUNT == 0:
    print("No images found, exiting...")
    exit()

if IMG_CHANNELS == 'Grayscale' or IMG_CHANNELS == 'Binarize':
    COLOR_CHANNELS = 1
else:
    COLOR_CHANNELS = len(IMG_CHANNELS)

RED = "\033[91m"
GREEN = "\033[92m"
DARK_GREY = "\033[90m"
NORMAL = "\033[0m"
def timestamp():
    return DARK_GREY + f"[{datetime.datetime.now().strftime('%H:%M:%S')}] " + NORMAL

print("\n----------------------------------------------\n")

print(timestamp() + f"Using {str(DEVICE).upper()} for training")
print(timestamp() + 'Number of CPU cores:', multiprocessing.cpu_count())
print()
print(timestamp() + "Training settings:")
print(timestamp() + "> Epochs:", NUM_EPOCHS)
print(timestamp() + "> Batch size:", BATCH_SIZE)
print(timestamp() + "> Classes:", CLASSES)
print(timestamp() + "> Split size:", SPLIT_SIZE)
print(timestamp() + "> Bounding boxes:", BOUNDINGBOXES)
print(timestamp() + "> Images:", IMG_COUNT)
print(timestamp() + "> Image size:", IMG_SIZE)
print(timestamp() + "> Image channels:", IMG_CHANNELS)
print(timestamp() + "> Color channels:", COLOR_CHANNELS)
print(timestamp() + "> IOU threshold:", IOU_THRESHOLD)
print(timestamp() + "> Confidence threshold:", CONFIDENCE_THRESHOLD)
print(timestamp() + "> NMS across all classes:", NMS_ACROSS_ALL_CLASSES)
print(timestamp() + "> Learning rate:", LEARNING_RATE)
print(timestamp() + "> Max learning rate:", MAX_LEARNING_RATE)
print(timestamp() + "> Dataset split:", TRAIN_VAL_RATIO)
print(timestamp() + "> Number of workers:", NUM_WORKERS)
print(timestamp() + "> Patience:", PATIENCE)
print(timestamp() + "> Shuffle:", SHUFFLE)
print(timestamp() + "> Pin memory:", PIN_MEMORY)
print(timestamp() + "> Drop last:", DROP_LAST)


def intersection_over_union(boxes_preds=None, boxes_labels=None):
    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    return intersection / (box1_area + box2_area - intersection + 1e-6)

def non_max_suppression(bboxes=None):
    assert type(bboxes) == list
    bboxes = [box for box in bboxes if box[1] > CONFIDENCE_THRESHOLD]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []
    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [box for box in bboxes if ((True) if NMS_ACROSS_ALL_CLASSES else (box[0] != chosen_box[0])) or intersection_over_union(torch.tensor(chosen_box[2:]), torch.tensor(box[2:])) < IOU_THRESHOLD]
        bboxes_after_nms.append(chosen_box)
    return bboxes_after_nms

def mean_average_precision(pred_boxes=None, true_boxes=None):
    average_precisions = []
    epsilon = 1e-6
    for c in range(CLASSES):
        detections = []
        ground_truths = []
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        if total_true_bboxes == 0:
            continue
        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            best_iou = 0
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(torch.tensor(detection[3:]), torch.tensor(gt[3:]))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            if best_iou > IOU_THRESHOLD:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))
    return sum(average_precisions) / len(average_precisions)

def get_bboxes(loader=None, model=None):
    all_pred_boxes = []
    all_true_boxes = []

    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(bboxes[idx])

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > CONFIDENCE_THRESHOLD:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    return all_pred_boxes, all_true_boxes

def convert_cellboxes(predictions=None):
    predictions = predictions.to(DEVICE)
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, SPLIT_SIZE, SPLIT_SIZE, CLASSES + 10)
    bboxes1 = predictions[..., CLASSES + 1:CLASSES + 5]
    bboxes2 = predictions[..., CLASSES + 6:CLASSES + 10]
    scores = torch.cat((predictions[..., CLASSES].unsqueeze(0), predictions[..., CLASSES + 5].unsqueeze(0)), dim=0)
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(SPLIT_SIZE).repeat(batch_size, SPLIT_SIZE, 1).unsqueeze(-1).to(DEVICE)
    x = 1 / SPLIT_SIZE * (best_boxes[..., :1] + cell_indices)
    y = 1 / SPLIT_SIZE * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / SPLIT_SIZE * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :CLASSES].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., CLASSES], predictions[..., CLASSES + 5]).unsqueeze(-1)
    converted_preds = torch.cat((predicted_class, best_confidence, converted_bboxes), dim=-1)
    return converted_preds

def cellboxes_to_boxes(out=None):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], SPLIT_SIZE * SPLIT_SIZE, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(SPLIT_SIZE * SPLIT_SIZE):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def bbox_coverage_ratio(box):
    x1 = box[0] - box[2] / 2
    y1 = box[1] - box[3] / 2
    x2 = box[0] + box[2] / 2
    y2 = box[1] + box[3] / 2
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(1, x2)
    y2 = min(1, y2)
    area_in_image = max(0, x2 - x1) * max(0, y2 - y1)
    area_total = box[2] * box[3]
    if area_total == 0:
        return 0.0
    else:
        return round(float(area_in_image / area_total), 5)

def RandomCrop(img, bboxes, min_width=0.3, min_height=0.3):
    any_bbox_in_image = False
    while any_bbox_in_image == False:
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(int(img.size[1] * random.uniform(min_height, 1.0)), int(img.size[0] * random.uniform(min_width, 1.0))))
        img_width, img_height = img.size[0], img.size[1]
        new_img = transforms.functional.crop(img, i, j, h, w)
        new_bboxes = []
        for box in bboxes:
            bb_cx = (box[1] * img_width - j) / w
            bb_cy = (box[2] * img_height - i) / h
            bb_w = (box[3] * img_width) / w
            bb_h = (box[4] * img_height) / h
            if bbox_coverage_ratio((bb_cx, bb_cy, bb_w, bb_h)) > 0.2:
                new_bboxes.append(torch.tensor([box[0], bb_cx, bb_cy, bb_w, bb_h]))
                any_bbox_in_image = True
    return new_img, new_bboxes

class CustomDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.files = [f for f in os.listdir(DATA_PATH) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_name = self.files[index]
        img_path = os.path.join(DATA_PATH, img_name)
        label_path = os.path.join(DATA_PATH, img_name.replace('.jpg', '.txt'))
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
        image, boxes = self.transform(image, boxes)
        label_matrix = torch.zeros((SPLIT_SIZE, SPLIT_SIZE, CLASSES + BOUNDINGBOXES * 5))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            i, j = int(SPLIT_SIZE * y), int(SPLIT_SIZE * x)
            if i >= SPLIT_SIZE or j >= SPLIT_SIZE: continue
            x_cell, y_cell = SPLIT_SIZE * x - j, SPLIT_SIZE * y - i
            width_cell, height_cell = (width * SPLIT_SIZE, height * SPLIT_SIZE)
            if label_matrix[i, j, CLASSES] == 0:
                label_matrix[i, j, CLASSES] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, CLASSES + 1:CLASSES + 5] = box_coordinates
                label_matrix[i, j, class_label] = 1
        return image, label_matrix

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions=None, target=None):
        predictions = predictions.reshape(-1, SPLIT_SIZE, SPLIT_SIZE, CLASSES + BOUNDINGBOXES * 5)
        iou_b1 = intersection_over_union(predictions[..., CLASSES + 1:CLASSES + 5], target[..., CLASSES + 1:CLASSES + 5])
        iou_b2 = intersection_over_union(predictions[..., CLASSES + 6:CLASSES + 10], target[..., CLASSES + 1:CLASSES + 5])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., CLASSES].unsqueeze(3)
        box_predictions = exists_box * (
            (bestbox * predictions[..., CLASSES + 6:CLASSES + 10] + (1 - bestbox) * predictions[..., CLASSES + 1:CLASSES + 5])
        )
        box_targets = exists_box * target[..., CLASSES + 1:CLASSES + 5]
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )
        pred_box = (
            bestbox * predictions[..., CLASSES + 5:CLASSES + 6] + (1 - bestbox) * predictions[..., CLASSES:CLASSES + 1]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., CLASSES:CLASSES + 1]),
        )
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., CLASSES:CLASSES + 1], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., CLASSES:CLASSES + 1], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., CLASSES + 5:CLASSES + 6], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., CLASSES:CLASSES + 1], start_dim=1)
        )
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :CLASSES], end_dim=-2,),
            torch.flatten(exists_box * target[..., :CLASSES], end_dim=-2,),
        )
        loss = (self.lambda_coord * box_loss + object_loss + self.lambda_noobj * no_object_loss + class_loss)
        return loss

# Define the model
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.darknet = self._create_conv_layers()
        self.fcs = self._create_fcs()

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self):
        layers = []
        in_channels = COLOR_CHANNELS
        for x in ARCHITECTURE:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]
                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]
        return nn.Sequential(*layers)

    def _create_fcs(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * SPLIT_SIZE * SPLIT_SIZE, SPLIT_SIZE * SPLIT_SIZE * (CLASSES + BOUNDINGBOXES * 5)),
            nn.LeakyReLU(0.1),
            nn.Linear(SPLIT_SIZE * SPLIT_SIZE * (CLASSES + BOUNDINGBOXES * 5), SPLIT_SIZE * SPLIT_SIZE * (CLASSES + BOUNDINGBOXES * 5)),
        )

def main():
    # Initialize model
    model = NeuralNetwork().to(DEVICE)

    def get_model_size_mb(model):
        total_params = 0
        for param in model.parameters():
            total_params += np.prod(param.size())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        bytes_per_param = next(model.parameters()).element_size()
        model_size_mb = (total_params * bytes_per_param) / (1024 ** 2)
        return total_params, trainable_params, non_trainable_params, model_size_mb

    total_params, trainable_params, non_trainable_params, model_size_mb = get_model_size_mb(model)

    print()
    print(timestamp() + "Model properties:")
    print(timestamp() + f"> Total parameters: {total_params}")
    print(timestamp() + f"> Trainable parameters: {trainable_params}")
    print(timestamp() + f"> Non-trainable parameters: {non_trainable_params}")
    print(timestamp() + f"> Predicted model size: {model_size_mb:.2f}MB")

    print("\n----------------------------------------------\n")

    print(timestamp() + "Loading...")

    # Create tensorboard logs folder if it doesn't exist
    if not os.path.exists(f"{PATH}/Training/ObjectDetection/logs"):
        os.makedirs(f"{PATH}/Training/ObjectDetection/logs")

    # Delete previous tensorboard logs
    for obj in os.listdir(f"{PATH}/Training/ObjectDetection/logs"):
        try:
            shutil.rmtree(f"{PATH}/Training/ObjectDetection/logs/{obj}")
        except:
            os.remove(f"{PATH}/Training/ObjectDetection/logs/{obj}")

    # Tensorboard setup
    summary_writer = SummaryWriter(f"{PATH}/Training/ObjectDetection/logs", comment="ObjectDetection-Training", flush_secs=20)

    # Transformations
    class Compose(object):
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, img, bboxes):
            for t in self.transforms:
                if t == RandomCrop:
                    img, bboxes = RandomCrop(img, bboxes)
                else:
                    img, bboxes = t(img), bboxes
            return img, bboxes

    transform = Compose([
        RandomCrop,
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # Create dataset
    dataset = CustomDataset()

    # Split the dataset into training and validation sets
    train_size = int(TRAIN_VAL_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataset.dataset.transform = transform

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=DROP_LAST)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=DROP_LAST)

    # Initialize loss function, optimizer and scheduler
    loss_fn = Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LEARNING_RATE, steps_per_epoch=len(train_dataloader), epochs=NUM_EPOCHS)

    # Early stopping variables
    best_validation_loss = float('inf')
    best_model = None
    best_model_epoch = None
    best_model_training_loss = None
    best_model_validation_loss = None
    wait = 0

    print(f"\r{timestamp()}Starting training...                       ")
    print("\n-----------------------------------------------------------------------------------------------------------------------------------\n")

    training_time_prediction = time.time()
    training_start_time = time.time()
    epoch_total_time = 0
    training_loss = 0
    validation_loss = 0
    training_mAP = 0
    validation_mAP = 0
    training_epoch = 0

    global PROGRESS_PRINT
    PROGRESS_PRINT = "initializing"
    def training_progress_print():
        global PROGRESS_PRINT
        def num_to_str(num: int):
            str_num = format(num, '.8f')
            while len(str_num) > 8:
                str_num = str_num[:-1]
            while len(str_num) < 8:
                str_num = str_num + '0'
            return str_num
        while PROGRESS_PRINT == "initializing":
            time.sleep(1)
        last_message = ""
        while PROGRESS_PRINT == "running":
            progress = (time.time() - epoch_total_start_time) / epoch_total_time
            if progress > 1: progress = 1
            if progress < 0: progress = 0
            progress = '█' * int(progress * 10) + '░' * (10 - int(progress * 10))
            epoch_time = round(epoch_total_time, 2) if epoch_total_time > 1 else round((epoch_total_time) * 1000)
            eta = time.strftime('%H:%M:%S', time.gmtime(round((training_time_prediction - training_start_time) / (training_epoch + 1) * NUM_EPOCHS - (training_time_prediction - training_start_time) + (training_time_prediction - time.time()), 2)))
            message = f"{progress} Epoch {training_epoch+1}, Train Loss: {num_to_str(training_loss)}, Val Loss: {num_to_str(validation_loss)}, Train mAP: {num_to_str(training_mAP)}, Val mAP: {num_to_str(validation_mAP)}, {epoch_time}{'s' if epoch_total_time > 1 else 'ms'}/Epoch, ETA: {eta}"
            print(f"\r{message}" + (" " * (len(last_message) - len(message)) if len(last_message) > len(message) else ""), end='', flush=True)
            last_message = message
            time.sleep(1)
        if PROGRESS_PRINT == "early stopped":
            print(f"\rEarly stopping at Epoch {training_epoch+1}, Train Loss: {num_to_str(training_loss)}, Val Loss: {num_to_str(validation_loss)}                                              ", end='', flush=True)
        elif PROGRESS_PRINT == "finished":
            print(f"\rFinished at Epoch {training_epoch+1}, Train Loss: {num_to_str(training_loss)}, Val Loss: {num_to_str(validation_loss)}                                              ", end='', flush=True)
        PROGRESS_PRINT = "received"
    threading.Thread(target=training_progress_print, daemon=True).start()

    for epoch in range(NUM_EPOCHS):
        epoch_total_start_time = time.time()


        epoch_training_start_time = time.time()

        # Training phase
        model.train()
        running_training_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            with torch.set_grad_enabled(True):
                inputs, labels = data[0].to(DEVICE, non_blocking=True), data[1].to(DEVICE, non_blocking=True)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                running_training_loss += loss.item()
        running_training_loss /= len(train_dataloader)
        training_loss = running_training_loss

        epoch_training_time = time.time() - epoch_training_start_time


        epoch_validation_start_time = time.time()

        # Validation phase
        model.eval()
        running_validation_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_dataloader, 0):
                inputs, labels = data[0].to(DEVICE, non_blocking=True), data[1].to(DEVICE, non_blocking=True)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                running_validation_loss += loss.item()

            frame = np.zeros((IMG_SIZE * 2, IMG_SIZE * 2, 3), dtype=np.uint8)
            random_indices = random.sample(range(len(train_dataloader.dataset)), 4)
            for i, idx in enumerate(random_indices):
                x, y = train_dataloader.dataset[idx]
                x = x.unsqueeze(0).to(DEVICE)
                bboxes = cellboxes_to_boxes(model(x))
                bboxes = non_max_suppression(bboxes[0])
                image = x[0].permute(1,2,0).to(DEVICE)
                image = (image * 255).byte().detach().cpu().numpy()
                row = i // 2
                col = i % 2
                frame[row*IMG_SIZE:(row+1)*IMG_SIZE, col*IMG_SIZE:(col+1)*IMG_SIZE] = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                for box in bboxes:
                    box = box[2:]
                    upper_left_x = (box[0] - box[2] / 2) * IMG_SIZE
                    upper_left_y = (box[1] - box[3] / 2) * IMG_SIZE
                    width = box[2] * IMG_SIZE
                    height = box[3] * IMG_SIZE
                    cv2.rectangle(frame, (int(col*IMG_SIZE + upper_left_x), int(row*IMG_SIZE + upper_left_y)), (int(col*IMG_SIZE + upper_left_x + width), int(row*IMG_SIZE + upper_left_y + height)), (255, 0, 0), 2)
                true_boxes = cellboxes_to_boxes(y.unsqueeze(0))[0]
                for box in true_boxes:
                    class_label, confidence, x, y, width, height = box
                    if confidence > 0.5:
                        upper_left_x = (x - width / 2) * IMG_SIZE
                        upper_left_y = (y - height / 2) * IMG_SIZE
                        width *= IMG_SIZE
                        height *= IMG_SIZE
                        cv2.rectangle(frame, (int(col*IMG_SIZE + upper_left_x), int(row*IMG_SIZE + upper_left_y)), (int(col*IMG_SIZE + upper_left_x + width), int(row*IMG_SIZE + upper_left_y + height)), (0, 255, 0), 2)

        running_validation_loss /= len(val_dataloader)
        validation_loss = running_validation_loss

        epoch_validation_time = time.time() - epoch_validation_start_time


        epoch_mAP_start_time = time.time()

        pred_boxes, target_boxes = get_bboxes(train_dataloader, model)
        training_mAP = mean_average_precision(pred_boxes=pred_boxes, true_boxes=target_boxes)
        pred_boxes, target_boxes = get_bboxes(val_dataloader, model)
        validation_mAP = mean_average_precision(pred_boxes=pred_boxes, true_boxes=target_boxes)

        epoch_mAP_time = time.time() - epoch_mAP_start_time


        # Early stopping
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model = model
            best_model_epoch = epoch
            best_model_training_loss = training_loss
            best_model_validation_loss = validation_loss
            best_model_training_mAP = training_mAP
            best_model_validation_mAP = validation_mAP
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                epoch_total_time = time.time() - epoch_total_start_time
                # Log values to Tensorboard
                summary_writer.add_scalars(f'Stats', {
                    'training_loss': training_loss,
                    'validation_loss': validation_loss,
                    'training_mAP': training_mAP,
                    'validation_mAP': validation_mAP,
                    'total_time': epoch_total_time,
                    'training_time': epoch_training_time,
                    'validation_time': epoch_validation_time,
                    'mAP_time': epoch_mAP_time
                }, epoch + 1)
                training_time_prediction = time.time()
                PROGRESS_PRINT = "early stopped"
                break

        epoch_total_time = time.time() - epoch_total_start_time

        # Log values to Tensorboard
        summary_writer.add_scalars(f'Stats', {
            'training_loss': training_loss,
            'validation_loss': validation_loss,
            'training_mAP': training_mAP,
            'validation_mAP': validation_mAP,
            'total_time': epoch_total_time,
            'training_time': epoch_training_time,
            'validation_time': epoch_validation_time,
            'mAP_time': epoch_mAP_time
        }, epoch + 1)
        summary_writer.add_images('Images', frame, epoch + 1, dataformats='HWC')
        training_epoch = epoch
        training_time_prediction = time.time()
        PROGRESS_PRINT = "running"

    if PROGRESS_PRINT != "early stopped":
        PROGRESS_PRINT = "finished"
    while PROGRESS_PRINT != "received":
        time.sleep(1)

    print("\n\n-----------------------------------------------------------------------------------------------------------------------------------")

    TRAINING_TIME = time.strftime('%H-%M-%S', time.gmtime(time.time() - training_start_time))
    TRAINING_DATE = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    print()
    print(timestamp() + f"Training completed after " + TRAINING_TIME.replace('-', ':'))

    # Save the last model
    print(timestamp() + "Saving the last model...")

    metadata = (f"epochs#{epoch+1}",
                f"batch#{BATCH_SIZE}",
                f"classes#{CLASSES}",
                f"outputs#{CLASSES}",
                f"image_count#{IMG_COUNT}",
                f"image_width#{IMG_SIZE}",
                f"image_height#{IMG_SIZE}",
                f"image_channels#{IMG_CHANNELS}",
                f"color_channels#{COLOR_CHANNELS}")
    metadata = {"data": metadata}
    metadata = {data: str(value).encode("ascii") for data, value in metadata.items()}

    last_model_saved = False
    for i in range(5):
        try:
            last_model = torch.jit.script(model)
            torch.jit.save(last_model, os.path.join(MODEL_PATH, f"ObjectDetectionModel-LAST-{TRAINING_DATE}.pt"), _extra_files=metadata)
            last_model_saved = True
            break
        except:
            print(timestamp() + "Failed to save the last model. Retrying...")
    print(timestamp() + "Last model saved successfully.") if last_model_saved else print(timestamp() + "Failed to save the last model.")

    # Save the best model
    print(timestamp() + "Saving the best model...")

    metadata = (f"epochs#{best_model_epoch+1}",
                f"batch#{BATCH_SIZE}",
                f"classes#{CLASSES}",
                f"outputs#{CLASSES}",
                f"image_count#{IMG_COUNT}",
                f"image_width#{IMG_SIZE}",
                f"image_height#{IMG_SIZE}",
                f"image_channels#{IMG_CHANNELS}",
                f"color_channels#{COLOR_CHANNELS}")
    metadata = {"data": metadata}
    metadata = {data: str(value).encode("ascii") for data, value in metadata.items()}

    best_model_saved = False
    for i in range(5):
        try:
            best_model = torch.jit.script(best_model)
            torch.jit.save(best_model, os.path.join(MODEL_PATH, f"ObjectDetectionModel-BEST-{TRAINING_DATE}.pt"), _extra_files=metadata)
            best_model_saved = True
            break
        except:
            print(timestamp() + "Failed to save the best model. Retrying...")
    print(timestamp() + "Best model saved successfully.") if best_model_saved else print(timestamp() + "Failed to save the best model.")

    print("\n----------------------------------------------\n")

if __name__ == '__main__':
    main()