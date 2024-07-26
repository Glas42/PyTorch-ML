import datetime
print(f"\n----------------------------------------------\n\n\033[90m[{datetime.datetime.now().strftime('%H:%M:%S')}] \033[0mImporting libraries...")

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
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
NUM_EPOCHS = 10000
BATCH_SIZE = 10
CLASSES = 10
IMG_WIDTH = 140
IMG_HEIGHT = 140
IMG_CHANNELS = ['Grayscale', 'Binarize', 'RGB', 'RG', 'GB', 'RB', 'R', 'G', 'B'][0]
LEARNING_RATE = 0.0001
MAX_LEARNING_RATE = 0.01
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
print(timestamp() + "> Image width:", IMG_WIDTH)
print(timestamp() + "> Image height:", IMG_HEIGHT)
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


def GenerateRandomSampleImage(model, dataset):
    random_index = random.randint(0, len(dataset) - 1)
    image, label = dataset[random_index][0].to(DEVICE, non_blocking=True), dataset[random_index][1].to(DEVICE, non_blocking=True)
    class_output, bbox_output = model(image.unsqueeze(0))
    class_prediction = class_output.tolist()[0]
    bbox_prediction = bbox_output.tolist()[0]
    image = image.permute(1, 2, 0).cpu().numpy()
    if image.shape[2] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    x1 = label[CLASSES + 0].item() * IMG_WIDTH - label[CLASSES + 2].item() * IMG_WIDTH / 2
    y1 = label[CLASSES + 1].item() * IMG_HEIGHT - label[CLASSES + 3].item() * IMG_HEIGHT / 2
    x2 = label[CLASSES + 0].item() * IMG_WIDTH + label[CLASSES + 2].item() * IMG_WIDTH / 2
    y2 = label[CLASSES + 1].item() * IMG_HEIGHT + label[CLASSES + 3].item() * IMG_HEIGHT / 2
    cv2.rectangle(image, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 1)
    x1 = bbox_prediction[0] * IMG_WIDTH - bbox_prediction[2] * IMG_WIDTH / 2
    y1 = bbox_prediction[1] * IMG_HEIGHT - bbox_prediction[3] * IMG_HEIGHT / 2
    x2 = bbox_prediction[0] * IMG_WIDTH + bbox_prediction[2] * IMG_WIDTH / 2
    y2 = bbox_prediction[1] * IMG_HEIGHT + bbox_prediction[3] * IMG_HEIGHT / 2
    cv2.rectangle(image, (round(x1), round(y1)), (round(x2), round(y2)), (255, 0, 0), 1)
    label_np = label.cpu().numpy()
    cv2.putText(image, str(np.argmax(label_np[:CLASSES])), (0, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.682, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(image, str(np.argmax(class_prediction)), (20, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.682, (255, 0, 0), 1, cv2.LINE_AA)
    return image

def IntersectionOverUnion(predictions=None, labels=None):
    box1_x1 = predictions[..., 0:1] - predictions[..., 2:3] / 2
    box1_y1 = predictions[..., 1:2] - predictions[..., 3:4] / 2
    box1_x2 = predictions[..., 0:1] + predictions[..., 2:3] / 2
    box1_y2 = predictions[..., 1:2] + predictions[..., 3:4] / 2
    box2_x1 = labels[..., 0:1] - labels[..., 2:3] / 2
    box2_y1 = labels[..., 1:2] - labels[..., 3:4] / 2
    box2_x2 = labels[..., 0:1] + labels[..., 2:3] / 2
    box2_y2 = labels[..., 1:2] + labels[..., 3:4] / 2
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union = box1_area + box2_area - intersection
    iou = torch.where(union == 0, torch.tensor(0.0, device=union.device), intersection / union)
    return iou

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

                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                img = img / 255.0

                if IMG_CHANNELS == 'Binarize':
                    img = cv2.threshold(img, 0.5, 1.0, cv2.THRESH_BINARY)[1]

                labels_file = os.path.join(DATA_PATH, file.replace(file.split(".")[-1], "txt"))
                if os.path.exists(labels_file):
                    with open(labels_file, 'r') as f:
                        content = str(f.read()).split(" ")
                        label = [0] * (CLASSES + 4)
                        label[int(content[0])] = 1
                        label[CLASSES + 0] = float(content[1])
                        label[CLASSES + 1] = float(content[2])
                        label[CLASSES + 2] = float(content[3])
                        label[CLASSES + 3] = float(content[4])
                    images.append(img)
                    labels.append(label)
                else:
                    pass

            if len(images) % round(len(files) / 100) == 0:
                print(f"\r{timestamp()}Caching {type} dataset... ({round(100 * len(images) / len(files))}%)", end='', flush=True)

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)

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
            image = self.transform(image)
            return image, torch.as_tensor(label, dtype=torch.float32)

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

            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img / 255.0

            if IMG_CHANNELS == 'Binarize':
                img = cv2.threshold(img, 0.5, 1.0, cv2.THRESH_BINARY)[1]

            with open(label_path, 'r') as f:
                content = str(f.read()).split(' ')
                label = [0] * (CLASSES + 4)
                label[int(content[0])] = 1
                label[CLASSES + 0] = float(content[1])
                label[CLASSES + 1] = float(content[2])
                label[CLASSES + 2] = float(content[3])
                label[CLASSES + 3] = float(content[4])

            image = np.array(img, dtype=np.float32)
            image = self.transform(image)
            return image, torch.as_tensor(label, dtype=torch.float32)

# Define the loss function
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.class_loss = nn.CrossEntropyLoss()
        self.bbox_loss = nn.L1Loss()

    def forward(self, predictions=None, target=None):
        class_output, bbox_output = predictions
        class_target, bbox_target = target[:, :CLASSES], target[:, CLASSES:]
        class_loss = self.class_loss(class_output, class_target.argmax(dim=1))
        bbox_loss = self.bbox_loss(bbox_output, bbox_target)
        return class_loss + bbox_loss

# Define the model
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=COLOR_CHANNELS, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))

        self.class_fc1 = nn.Linear(in_features=128, out_features=240)
        self.class_fc2 = nn.Linear(in_features=240, out_features=120)
        self.class_out = nn.Linear(in_features=120, out_features=CLASSES)

        self.box_fc1 = nn.Linear(in_features=128, out_features=240)
        self.box_fc2 = nn.Linear(in_features=240, out_features=120)
        self.box_out = nn.Linear(in_features=120, out_features=4)

    def forward(self, t):
        t = self.conv1(t)
        t = nn.functional.relu(t)
        t = self.pool1(t)

        t = self.conv2(t)
        t = nn.functional.relu(t)
        t = self.pool2(t)

        t = self.conv3(t)
        t = nn.functional.relu(t)
        t = self.pool3(t)

        t = self.conv4(t)
        t = nn.functional.relu(t)
        t = self.pool4(t)

        t = torch.flatten(t, start_dim=1)

        class_t = self.class_fc1(t)
        class_t = nn.functional.relu(class_t)

        class_t = self.class_fc2(class_t)
        class_t = nn.functional.relu(class_t)

        class_t = nn.functional.softmax(self.class_out(class_t), dim=1)

        box_t = self.box_fc1(t)
        box_t = nn.functional.relu(box_t)

        box_t = self.box_fc2(box_t)
        box_t = nn.functional.relu(box_t)

        box_t = self.box_out(box_t)
        box_t = nn.functional.sigmoid(box_t)

        return [class_t, box_t]

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
    if not os.path.exists(f"{PATH}/Training/ClassificationLocalization/logs"):
        os.makedirs(f"{PATH}/Training/ClassificationLocalization/logs")

    # Delete previous tensorboard logs
    for obj in os.listdir(f"{PATH}/Training/ClassificationLocalization/logs"):
        try:
            shutil.rmtree(f"{PATH}/Training/ClassificationLocalization/logs/{obj}")
        except:
            os.remove(f"{PATH}/Training/ClassificationLocalization/logs/{obj}")

    # Tensorboard setup
    summary_writer = SummaryWriter(f"{PATH}/Training/ClassificationLocalization/logs", comment="ClassificationLocalization-Training", flush_secs=20)

    # Transformations
    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor()
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
    criterion = Loss()
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
            optimizer.zero_grad()
            with autocast():
                class_output, bbox_output = model(inputs)
                loss = criterion((class_output, bbox_output), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            if scale <= scaler.get_scale():
                scheduler.step()
            running_training_loss += loss.item()
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
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_validation_loss += loss.item()
        running_validation_loss /= len(val_dataloader)
        validation_loss = running_validation_loss

        epoch_validation_time = time.time() - epoch_validation_start_time


        epoch_mAP_start_time = time.time()

        # Calculate mAP on train dataset

        # Calculate mAP on validation dataset

        epoch_mAP_time = time.time() - epoch_mAP_start_time


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
                image = GenerateRandomSampleImage(model, val_dataset)
                summary_writer.add_image(f'Image', image, epoch, dataformats='HWC')
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
        image = GenerateRandomSampleImage(model, val_dataset)
        summary_writer.add_image(f'Validation Dataset Image', image, epoch, dataformats='HWC')
        image = GenerateRandomSampleImage(model, train_dataset)
        summary_writer.add_image(f'Train Dataset Image', image, epoch, dataformats='HWC')
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

    torch.cuda.empty_cache()

    model.eval()
    total_train = 0
    correct_train = 0
    with torch.no_grad():
        for data in train_dataloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == torch.argmax(labels, dim=1)).sum().item()
    training_dataset_accuracy = str(round(100 * (correct_train / total_train), 2)) + "%"

    torch.cuda.empty_cache()

    total_val = 0
    correct_val = 0
    with torch.no_grad():
        for data in val_dataloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == torch.argmax(labels, dim=1)).sum().item()
    validation_dataset_accuracy = str(round(100 * (correct_val / total_val), 2)) + "%"

    metadata_optimizer = str(optimizer).replace('\n', '')
    metadata_criterion = str(criterion).replace('\n', '')
    metadata_model = str(model).replace('\n', '')
    metadata = (f"epochs#{epoch}",
                f"batch#{BATCH_SIZE}",
                f"classes#{CLASSES}",
                f"outputs#{CLASSES}",
                f"image_count#{IMG_COUNT}",
                f"image_width#{IMG_WIDTH}",
                f"image_height#{IMG_HEIGHT}",
                f"image_channels#{IMG_CHANNELS}",
                f"color_channels#{COLOR_CHANNELS}",
                f"learning_rate#{LEARNING_RATE}",
                f"max_learning_rate#{MAX_LEARNING_RATE}",
                f"dataset_split#{TRAIN_VAL_RATIO}",
                f"number_of_workers#{NUM_WORKERS}",
                f"dropout#{DROPOUT}",
                f"patience#{PATIENCE}",
                f"shuffle#{SHUFFLE}",
                f"pin_memory#{PIN_MEMORY}",
                f"training_time#{TRAINING_TIME}",
                f"training_date#{TRAINING_DATE}",
                f"training_device#{DEVICE}",
                f"training_os#{os.name}",
                f"architecture#{metadata_model}",
                f"torch_version#{torch.__version__}",
                f"numpy_version#{np.__version__}",
                f"pil_version#{Image.__version__}",
                f"train_transform#{train_transform}",
                f"val_transform#{val_transform}",
                f"optimizer#{metadata_optimizer}",
                f"loss_function#{metadata_criterion}",
                f"training_size#{train_size}",
                f"validation_size#{val_size}",
                f"training_loss#{best_model_training_loss}",
                f"validation_loss#{best_model_validation_loss}",
                f"training_dataset_accuracy#{training_dataset_accuracy}",
                f"validation_dataset_accuracy#{validation_dataset_accuracy}")
    metadata = {"data": metadata}
    metadata = {data: str(value).encode("ascii") for data, value in metadata.items()}

    last_model_saved = False
    for i in range(5):
        try:
            last_model = torch.jit.script(model)
            torch.jit.save(last_model, os.path.join(MODEL_PATH, f"ClassificationLocalizationModel-LAST-{TRAINING_DATE}.pt"), _extra_files=metadata)
            last_model_saved = True
            break
        except:
            print(timestamp() + "Failed to save the last model. Retrying...")
    print(timestamp() + "Last model saved successfully.") if last_model_saved else print(timestamp() + "Failed to save the last model.")

    # Save the best model
    print(timestamp() + "Saving the best model...")

    torch.cuda.empty_cache()

    best_model.eval()
    total_train = 0
    correct_train = 0
    with torch.no_grad():
        for data in train_dataloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = best_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == torch.argmax(labels, dim=1)).sum().item()
    training_dataset_accuracy = str(round(100 * (correct_train / total_train), 2)) + "%"

    torch.cuda.empty_cache()

    total_val = 0
    correct_val = 0
    with torch.no_grad():
        for data in val_dataloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = best_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == torch.argmax(labels, dim=1)).sum().item()
    validation_dataset_accuracy = str(round(100 * (correct_val / total_val), 2)) + "%"

    metadata_optimizer = str(optimizer).replace('\n', '')
    metadata_criterion = str(criterion).replace('\n', '')
    metadata_model = str(best_model).replace('\n', '')
    metadata = (f"epochs#{best_model_epoch}",
                f"batch#{BATCH_SIZE}",
                f"classes#{CLASSES}",
                f"outputs#{CLASSES}",
                f"image_count#{IMG_COUNT}",
                f"image_width#{IMG_WIDTH}",
                f"image_height#{IMG_HEIGHT}",
                f"image_channels#{IMG_CHANNELS}",
                f"color_channels#{COLOR_CHANNELS}",
                f"learning_rate#{LEARNING_RATE}",
                f"max_learning_rate#{MAX_LEARNING_RATE}",
                f"dataset_split#{TRAIN_VAL_RATIO}",
                f"number_of_workers#{NUM_WORKERS}",
                f"dropout#{DROPOUT}",
                f"patience#{PATIENCE}",
                f"shuffle#{SHUFFLE}",
                f"pin_memory#{PIN_MEMORY}",
                f"training_time#{TRAINING_TIME}",
                f"training_date#{TRAINING_DATE}",
                f"training_device#{DEVICE}",
                f"training_os#{os.name}",
                f"architecture#{metadata_model}",
                f"torch_version#{torch.__version__}",
                f"numpy_version#{np.__version__}",
                f"pil_version#{Image.__version__}",
                f"train_transform#{train_transform}",
                f"val_transform#{val_transform}",
                f"optimizer#{metadata_optimizer}",
                f"loss_function#{metadata_criterion}",
                f"training_size#{train_size}",
                f"validation_size#{val_size}",
                f"training_loss#{training_loss}",
                f"validation_loss#{validation_loss}",
                f"training_dataset_accuracy#{training_dataset_accuracy}",
                f"validation_dataset_accuracy#{validation_dataset_accuracy}")
    metadata = {"data": metadata}
    metadata = {data: str(value).encode("ascii") for data, value in metadata.items()}

    best_model_saved = False
    for i in range(5):
        try:
            best_model = torch.jit.script(best_model)
            torch.jit.save(best_model, os.path.join(MODEL_PATH, f"ClassificationLocalizationModel-BEST-{TRAINING_DATE}.pt"), _extra_files=metadata)
            best_model_saved = True
            break
        except:
            print(timestamp() + "Failed to save the best model. Retrying...")
    print(timestamp() + "Best model saved successfully.") if best_model_saved else print(timestamp() + "Failed to save the best model.")

    print("\n----------------------------------------------\n")

if __name__ == '__main__':
    main()