import datetime
print(f"\n----------------------------------------------\n\n\033[90m[{datetime.datetime.now().strftime('%H:%M:%S')}] \033[0mImporting libraries...")

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from collections import Counter
import torch.optim as optim
import albumentations as A
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
NUM_EPOCHS = 100
BATCH_SIZE = 4
CLASSES = 10
SPLIT_SIZE = 7
BOUNDINGBOXES = 2
IMG_SIZE = 448
IMG_CHANNELS = ['Grayscale', 'Binarize', 'RGB', 'RG', 'GB', 'RB', 'R', 'G', 'B'][2]
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5
NMS_ACROSS_ALL_CLASSES = True
LEARNING_RATE = 0.001
MAX_LEARNING_RATE = 0.001
TRAIN_VAL_RATIO = 0.8
NUM_WORKERS = 0
DROPOUT = 0.1
PATIENCE = -1
SHUFFLE = True
PIN_MEMORY = False
DROP_LAST = True
CACHE = True

IMG_COUNT = 0
for file in os.listdir(DATA_PATH):
    if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
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
print(timestamp() + "> Images:", IMG_COUNT)
print(timestamp() + "> Image size:", IMG_SIZE)
print(timestamp() + "> Image channels:", IMG_CHANNELS)
print(timestamp() + "> Color channels:", COLOR_CHANNELS)
print(timestamp() + "> Learning rate:", LEARNING_RATE)
print(timestamp() + "> Max learning rate:", MAX_LEARNING_RATE)
print(timestamp() + "> Dataset split:", TRAIN_VAL_RATIO)
print(timestamp() + "> Number of workers:", NUM_WORKERS)
print(timestamp() + "> Dropout:", DROPOUT)
print(timestamp() + "> Patience:", PATIENCE)
print(timestamp() + "> Shuffle:", SHUFFLE)
print(timestamp() + "> Pin memory:", PIN_MEMORY)
print(timestamp() + "> Drop last:", DROP_LAST)
print(timestamp() + "> Cache:", CACHE)


# Custom dataset class
if CACHE:
    def load_data(files=None, type=None):
        images = []
        labels = []
        print(f"\r{timestamp()}Caching {type} dataset...           ", end='', flush=True)
        for file in os.listdir(DATA_PATH):
            if file in files:
                if IMG_CHANNELS== 'Grayscale' or IMG_CHANNELS == 'Binarize':
                    img = Image.open(os.path.join(DATA_PATH, file)).convert('L')  # Convert to grayscale
                    img = np.array(img)
                else:
                    img = Image.open(os.path.join(DATA_PATH, file))
                    img = np.array(img)

                    if IMG_CHANNELS == 'RG':
                        img = np.stack((img[:, :, 0], img[:, :, 1]), axis=2)
                    elif IMG_CHANNELS == 'GB':
                        img = np.stack((img[:, :, 1], img[:, :, 2]), axis=2)
                    elif IMG_CHANNELS == 'RB':
                        img = np.stack((img[:, :, 0], img[:, :, 2]), axis=2)
                    elif IMG_CHANNELS == 'R':
                        img = img[:, :, 0]
                        img = np.expand_dims(img, axis=2)
                    elif IMG_CHANNELS == 'G':
                        img = img[:, :, 1]
                        img = np.expand_dims(img, axis=2)
                    elif IMG_CHANNELS == 'B':
                        img = img[:, :, 2]
                        img = np.expand_dims(img, axis=2)

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0

                if IMG_CHANNELS == 'Binarize':
                    img = cv2.threshold(img, 0.5, 1.0, cv2.THRESH_BINARY)[1]

                labels_file = os.path.join(DATA_PATH, file.replace(file.split(".")[-1], "txt"))
                if os.path.exists(labels_file):
                    boxes = []
                    with open(labels_file) as f:
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
                            boxes.append([x, y, width, height, class_label])
                    boxes = torch.tensor(boxes)
                    images.append(img)
                    labels.append(boxes)
                else:
                    pass

            if len(images) % round(len(files) / 100) if round(len(files) / 100) > 0 else True:
                print(f"\r{timestamp()}Caching {type} dataset... ({round(100 * len(images) / len(files))}%)", end='', flush=True)

        return np.array(images, dtype=np.float32), labels

    class CustomDataset(Dataset):
        def __init__(self, images, labels, transform=None):
            self.images = images
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = self.images[idx]
            label = self.labels[idx]
            transformed = self.transform(image=image, bboxes=label)
            image = transformed["image"]
            label = transformed["bboxes"]
            return image, label

else:

    class CustomDataset(Dataset):
        def __init__(self, files=None, transform=None):
            self.files = files
            self.transform = transform

        def __len__(self):
            return len(self.files)

        def __getitem__(self, index):
            image_name = self.files[index]
            image_path = os.path.join(DATA_PATH, image_name)
            label_path = os.path.join(DATA_PATH, image_name.replace(image_name.split('.')[-1], 'txt'))

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
                    boxes.append([x, y, width, height, class_label])
            boxes = torch.tensor(boxes)

            if IMG_CHANNELS== 'Grayscale' or IMG_CHANNELS == 'Binarize':
                img = Image.open(image_path).convert('L')
                img = np.array(img)
            else:
                img = Image.open(image_path)
                img = np.array(img)

                if IMG_CHANNELS == 'RG':
                    img = np.stack((img[:, :, 0], img[:, :, 1]), axis=2)
                elif IMG_CHANNELS == 'GB':
                    img = np.stack((img[:, :, 1], img[:, :, 2]), axis=2)
                elif IMG_CHANNELS == 'RB':
                    img = np.stack((img[:, :, 0], img[:, :, 2]), axis=2)
                elif IMG_CHANNELS == 'R':
                    img = img[:, :, 0]
                    img = np.expand_dims(img, axis=2)
                elif IMG_CHANNELS == 'G':
                    img = img[:, :, 1]
                    img = np.expand_dims(img, axis=2)
                elif IMG_CHANNELS == 'B':
                    img = img[:, :, 2]
                    img = np.expand_dims(img, axis=2)

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0

            if IMG_CHANNELS == 'Binarize':
                img = cv2.threshold(img, 0.5, 1.0, cv2.THRESH_BINARY)[1]

            if self.transform != None:
                transformed = self.transform(image=image, bboxes=label)
                image = transformed["image"]
                label = transformed["bboxes"]
            else:
                image = transforms.ToTensor()(image)
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

# Define the model
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        x = self.linear(x)
        return x

def main():
    # Initialize model
    model = ConvolutionalNeuralNetwork().to(DEVICE)

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
    train_transform = A.Compose([
        A.RandomCrop(width=round(random.uniform(0.5, 1) * IMG_SIZE), height=round(random.uniform(0.5, 1) * IMG_SIZE)),
        A.Rotate(limit=15, p=1),
        A.Resize(IMG_SIZE, IMG_SIZE)
    ], bbox_params=A.BboxParams(format='yolo'))

    val_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE)
    ])

    # Create datasets
    all_files = [f for f in os.listdir(DATA_PATH) if (f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")) and os.path.exists(f"{DATA_PATH}/{f.replace(f.split('.')[-1], 'txt')}")]
    random.shuffle(all_files)
    train_size = int(len(all_files) * TRAIN_VAL_RATIO)
    val_size = len(all_files) - train_size
    train_files = all_files[:train_size]
    val_files = all_files[train_size:]

    if CACHE:
        train_images, train_labels = load_data(train_files, "train")
        val_images, val_labels = load_data(val_files, "val")
        train_dataset = CustomDataset(train_images, train_labels, transform=train_transform)
        val_dataset = CustomDataset(val_images, val_labels, transform=val_transform)
    else:
        train_dataset = CustomDataset(train_files, transform=train_transform)
        val_dataset = CustomDataset(val_files, transform=val_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=DROP_LAST)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=DROP_LAST)

    # Initialize scaler, loss function, optimizer and scheduler
    scaler = GradScaler()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LEARNING_RATE, steps_per_epoch=len(train_dataloader), epochs=NUM_EPOCHS)

    # Early stopping variables
    best_validation_loss = float('inf')
    best_model = None
    best_model_epoch = None
    best_model_training_loss = None
    best_model_validation_loss = None
    wait = 0

    print(f"\r{timestamp()}Starting training...                ")
    print("\n-----------------------------------------------------------------------------------------------------------\n")

    training_time_prediction = time.time()
    training_start_time = time.time()
    epoch_total_time = 0
    training_loss = 0
    validation_loss = 0
    training_epoch = 0

    global PROGRESS_PRINT
    PROGRESS_PRINT = "initializing"
    def training_progress_print():
        global PROGRESS_PRINT
        def num_to_str(num: int):
            str_num = format(num, '.15f')
            while len(str_num) > 15:
                str_num = str_num[:-1]
            while len(str_num) < 15:
                str_num = str_num + '0'
            return str_num
        while PROGRESS_PRINT == "initializing":
            time.sleep(1)
        last_message = ""
        while PROGRESS_PRINT == "running":
            progress = (time.time() - epoch_total_start_time) / epoch_total_time
            if progress > 1: progress = 1
            if progress < 0: progress = 0
            progress = '█' * round(progress * 10) + '░' * (10 - round(progress * 10))
            epoch_time = round(epoch_total_time, 2) if epoch_total_time > 1 else round((epoch_total_time) * 1000)
            eta = time.strftime('%H:%M:%S', time.gmtime(round((training_time_prediction - training_start_time) / (training_epoch) * NUM_EPOCHS - (training_time_prediction - training_start_time) + (training_time_prediction - time.time()), 2)))
            message = f"{progress} Epoch {training_epoch}, Train Loss: {num_to_str(training_loss)}, Val Loss: {num_to_str(validation_loss)}, {epoch_time}{'s' if epoch_total_time > 1 else 'ms'}/Epoch, ETA: {eta}"
            print(f"\r{message}" + (" " * (len(last_message) - len(message)) if len(last_message) > len(message) else ""), end='', flush=True)
            last_message = message
            time.sleep(1)
        if PROGRESS_PRINT == "early stopped":
            message = f"Early stopping at Epoch {training_epoch}, Train Loss: {num_to_str(training_loss)}, Val Loss: {num_to_str(validation_loss)}"
            print(f"\r{message}" + (" " * (len(last_message) - len(message)) if len(last_message) > len(message) else ""), end='', flush=True)
        elif PROGRESS_PRINT == "finished":
            message = f"Finished at Epoch {training_epoch}, Train Loss: {num_to_str(training_loss)}, Val Loss: {num_to_str(validation_loss)}"
            print(f"\r{message}" + (" " * (len(last_message) - len(message)) if len(last_message) > len(message) else ""), end='', flush=True)
        PROGRESS_PRINT = "received"
    threading.Thread(target=training_progress_print, daemon=True).start()

    for epoch, _ in enumerate(range(NUM_EPOCHS), 1):
        epoch_total_start_time = time.time()


        epoch_training_start_time = time.time()

        # Training phase
        model.train()
        running_training_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data[0].to(DEVICE, non_blocking=True), data[1].to(DEVICE, non_blocking=True)

        running_training_loss /= len(train_dataloader)
        training_loss = running_training_loss

        epoch_training_time = time.time() - epoch_training_start_time


        epoch_validation_start_time = time.time()

        # Validation phase
        model.eval()
        running_validation_loss = 0.0
        with torch.no_grad(), autocast():
            for i, data in enumerate(val_dataloader, 0):
                inputs, labels = data[0].to(DEVICE, non_blocking=True), data[1].to(DEVICE, non_blocking=True)

        running_validation_loss /= len(val_dataloader)
        validation_loss = running_validation_loss

        epoch_validation_time = time.time() - epoch_validation_start_time


        # Early stopping
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model = model
            best_model_epoch = epoch
            best_model_training_loss = training_loss
            best_model_validation_loss = validation_loss
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE and PATIENCE > 0:
                epoch_total_time = time.time() - epoch_total_start_time
                # Log values to Tensorboard
                summary_writer.add_scalars(f'Stats', {
                    'train_loss': training_loss,
                    'validation_loss': validation_loss,
                    'epoch_total_time': epoch_total_time,
                    'epoch_training_time': epoch_training_time,
                    'epoch_validation_time': epoch_validation_time
                }, epoch)
                training_time_prediction = time.time()
                PROGRESS_PRINT = "early stopped"
                break

        epoch_total_time = time.time() - epoch_total_start_time

        # Log values to Tensorboard
        summary_writer.add_scalars(f'Stats', {
            'train_loss': training_loss,
            'validation_loss': validation_loss,
            'epoch_total_time': epoch_total_time,
            'epoch_training_time': epoch_training_time,
            'epoch_validation_time': epoch_validation_time
        }, epoch)
        training_epoch = epoch
        training_time_prediction = time.time()
        PROGRESS_PRINT = "running"

    if PROGRESS_PRINT != "early stopped":
        PROGRESS_PRINT = "finished"
    while PROGRESS_PRINT != "received":
        time.sleep(1)

    print("\n\n-----------------------------------------------------------------------------------------------------------")

    TRAINING_TIME = time.strftime('%H-%M-%S', time.gmtime(time.time() - training_start_time))
    TRAINING_DATE = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    print()
    print(timestamp() + f"Training completed after " + TRAINING_TIME.replace('-', ':'))

    # Save the last model
    print(timestamp() + "Saving the last model...")

    metadata = (f"epochs#{epoch}",
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

    metadata = (f"epochs#{best_model_epoch}",
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