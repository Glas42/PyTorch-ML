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
NUM_EPOCHS = 100
BATCH_SIZE = 500
CLASSES = 4
IMG_WIDTH = 90
IMG_HEIGHT = 150
IMG_CHANNELS = ['Grayscale', 'Binarize', 'RGB', 'RG', 'GB', 'RB', 'R', 'G', 'B'][0]
LEARNING_RATE = 0.001
MAX_LEARNING_RATE = 0.001
TRAIN_VAL_RATIO = 0.8
NUM_WORKERS = 0
DROPOUT = 0.1
PATIENCE = 10
SHUFFLE = True
PIN_MEMORY = False
DROP_LAST = True

IMG_COUNT = 0
for file in os.listdir(DATA_PATH):
    if file.endswith(".png"):
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


# Custom dataset class
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
            content = str(f.read())
            if content.isdigit() and 0 <= int(content) < CLASSES:
                user_input = [0] * CLASSES
                user_input[int(content)] = 1

        image = np.array(img, dtype=np.float32)
        image = self.transform(image)
        return image, torch.as_tensor(user_input, dtype=torch.float32)

# Define the model
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv2d_1 = nn.Conv2d(COLOR_CHANNELS, 32, (3, 3), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu_1 = nn.ReLU()
        self.maxpool_1 = nn.MaxPool2d((2, 2))

        self.conv2d_2 = nn.Conv2d(32, 64, (3, 3), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu_2 = nn.ReLU()
        self.maxpool_2 = nn.MaxPool2d((2, 2))

        self.conv2d_3 = nn.Conv2d(64, 128, (3, 3), padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu_3 = nn.ReLU()
        self.maxpool_3 = nn.MaxPool2d((2, 2))

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(DROPOUT)
        self.linear_1 = nn.Linear(128 * (IMG_WIDTH // 8) * (IMG_HEIGHT // 8), 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu_4 = nn.ReLU()
        self.linear_2 = nn.Linear(256, CLASSES, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.bn1(x)
        x = self.relu_1(x)
        x = self.maxpool_1(x)

        x = self.conv2d_2(x)
        x = self.bn2(x)
        x = self.relu_2(x)
        x = self.maxpool_2(x)

        x = self.conv2d_3(x)
        x = self.bn3(x)
        x = self.relu_3(x)
        x = self.maxpool_3(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear_1(x)
        x = self.bn4(x)
        x = self.relu_4(x)
        x = self.linear_2(x)
        x = self.softmax(x)
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
    if not os.path.exists(f"{PATH}/Training/Classification/logs"):
        os.makedirs(f"{PATH}/Training/Classification/logs")

    # Delete previous tensorboard logs
    for obj in os.listdir(f"{PATH}/Training/Classification/logs"):
        try:
            shutil.rmtree(f"{PATH}/Training/Classification/logs/{obj}")
        except:
            os.remove(f"{PATH}/Training/Classification/logs/{obj}")

    # Tensorboard setup
    summary_writer = SummaryWriter(f"{PATH}/Training/Classification/logs", comment="Classification-Training", flush_secs=20)

    # Transformations
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(35),
        transforms.RandomCrop((round(IMG_HEIGHT * random.uniform(0.5, 1)), round(IMG_WIDTH * random.uniform(0.5, 1)))),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH))
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

    train_dataset = CustomDataset(train_files, transform=train_transform)
    val_dataset = CustomDataset(val_files, transform=val_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=DROP_LAST)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=DROP_LAST)

    # Initialize scaler, loss function, optimizer and scheduler
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()
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
            progress = '█' * int(progress * 10) + '░' * (10 - int(progress * 10))
            epoch_time = round(epoch_total_time, 2) if epoch_total_time > 1 else round((epoch_total_time) * 1000)
            eta = time.strftime('%H:%M:%S', time.gmtime(round((training_time_prediction - training_start_time) / (training_epoch + 1) * NUM_EPOCHS - (training_time_prediction - training_start_time) + (training_time_prediction - time.time()), 2)))
            message = f"{progress} Epoch {training_epoch+1}, Train Loss: {num_to_str(training_loss)}, Val Loss: {num_to_str(validation_loss)}, {epoch_time}{'s' if epoch_total_time > 1 else 'ms'}/Epoch, ETA: {eta}"
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
            inputs, labels = data[0].to(DEVICE, non_blocking=True), data[1].to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
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
            if wait >= PATIENCE:
                epoch_total_time = time.time() - epoch_total_start_time
                # Log values to Tensorboard
                summary_writer.add_scalars(f'Stats', {
                    'train_loss': training_loss,
                    'validation_loss': validation_loss,
                    'epoch_total_time': epoch_total_time,
                    'epoch_training_time': epoch_training_time,
                    'epoch_validation_time': epoch_validation_time
                }, epoch + 1)
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
        }, epoch + 1)
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
    metadata = (f"epochs#{epoch+1}",
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
            torch.jit.save(last_model, os.path.join(MODEL_PATH, f"ClassificationModel-LAST-{TRAINING_DATE}.pt"), _extra_files=metadata)
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
    metadata = (f"epochs#{best_model_epoch+1}",
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
            torch.jit.save(best_model, os.path.join(MODEL_PATH, f"ClassificationModel-BEST-{TRAINING_DATE}.pt"), _extra_files=metadata)
            best_model_saved = True
            break
        except:
            print(timestamp() + "Failed to save the best model. Retrying...")
    print(timestamp() + "Best model saved successfully.") if best_model_saved else print(timestamp() + "Failed to save the best model.")

    print("\n----------------------------------------------\n")

if __name__ == '__main__':
    main()