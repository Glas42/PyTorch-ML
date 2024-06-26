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
BATCH_SIZE = 100
NUM_CLASSES = 10
NUM_BBOX = 4
IMG_WIDTH = 252
IMG_HEIGHT = 252
IMG_CHANNELS = ['Grayscale', 'Binarize', 'RGB', 'RG', 'GB', 'RB', 'R', 'G', 'B'][0]
LEARNING_RATE = 0.001
MAX_LEARNING_RATE = 0.001
TRAIN_VAL_RATIO = 0.8
NUM_WORKERS = 0
DROPOUT = 0.1
PATIENCE = 10
SHUFFLE = True
PIN_MEMORY = False

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
print(timestamp() + "> Classes:", NUM_CLASSES)
print(timestamp() + "> Bounding boxes:", NUM_BBOX)
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


def load_data():
    images = []
    user_inputs = []
    print(f"\r{timestamp()}Loading dataset...", end='', flush=True)
    for file in os.listdir(DATA_PATH):
        if file.endswith(".png"):
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

            bbox_file = os.path.join(DATA_PATH, file.replace(".png", ".txt"))
            if os.path.exists(bbox_file):
                with open(bbox_file, 'r') as f:
                    content = f.read().strip().split('\n')
                    for line in content:
                        class_id, center_x, center_y, width, height = [float(x) for x in line.split(',')]
                        bbox = np.array([class_id, center_x, center_y, width, height])
                        user_inputs.append(bbox)
            else:
                pass

        if len(images) % 100 == 0:
            print(f"\r{timestamp()}Loading dataset... ({round(100 * len(images) / IMG_COUNT)}%)", end='', flush=True)

    return np.array(images, dtype=np.float32), np.array(user_inputs, dtype=np.float32)

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, images, user_inputs, transform=None):
        self.images = images
        self.user_inputs = user_inputs
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        user_input = self.user_inputs[idx]
        image = self.transform(image)
        return image, torch.as_tensor(user_input, dtype=torch.float32)

# Define the model
class ObjectDetectionModel(nn.Module):
    def __init__(self):
        super(ObjectDetectionModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(COLOR_CHANNELS, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(256 * (IMG_HEIGHT // 16) * (IMG_WIDTH // 16), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(512, NUM_CLASSES + NUM_BBOX),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        class_scores = output[:, :NUM_CLASSES]
        bbox_outputs = output[:, NUM_CLASSES:]
        return class_scores, bbox_outputs

def main():
    # Initialize model
    model = ObjectDetectionModel().to(DEVICE)

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

    # Load data
    images, user_inputs = load_data()

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(35),
        transforms.RandomCrop((round(IMG_HEIGHT * random.uniform(0.5, 1)), round(IMG_WIDTH * random.uniform(0.5, 1)))),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH))
    ])

    # Create dataset
    dataset = CustomDataset(images, user_inputs, transform=transform)

    # Split the dataset into training and validation sets
    train_size = int(TRAIN_VAL_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # Initialize scaler, loss function, optimizer and scheduler
    scaler = GradScaler()
    class_criterion = nn.CrossEntropyLoss()
    bbox_criterion = nn.MSELoss()
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
        while PROGRESS_PRINT == "running":

            progress = (time.time() - epoch_total_start_time) / epoch_total_time
            if progress > 1: progress = 1
            if progress < 0: progress = 0

            progress = '█' * int(progress * 10) + '░' * (10 - int(progress * 10))
            epoch_time = round(epoch_total_time, 2) if epoch_total_time > 1 else round((epoch_total_time) * 1000)
            eta = time.strftime('%H:%M:%S', time.gmtime(round((training_time_prediction - training_start_time) / (training_epoch + 1) * NUM_EPOCHS - (training_time_prediction - training_start_time) + (training_time_prediction - time.time()), 2)))

            print(f"\r{progress} Epoch {training_epoch+1}, Train Loss: {num_to_str(training_loss)}, Val Loss: {num_to_str(validation_loss)}, {epoch_time}{'s' if epoch_total_time > 1 else 'ms'}/Epoch, ETA: {eta}                       ", end='', flush=True)

            time.sleep(epoch_total_time/10 if epoch_total_time/10 >= 0.1 else 0.1)
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
        for data in train_dataloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                class_scores, bbox_outputs = model(images)
                class_loss = class_criterion(class_scores, labels[:, 0].long())
                bbox_loss = bbox_criterion(bbox_outputs, labels[:, 1:])
                loss = class_loss + bbox_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_training_loss += loss.item()
        running_training_loss /= len(train_dataloader)
        training_loss = running_training_loss

        epoch_training_time = time.time() - epoch_training_start_time


        epoch_validation_start_time = time.time()

        # Validation phase
        model.eval()
        running_validation_loss = 0.0
        for data in val_dataloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.no_grad():
                class_scores, bbox_outputs = model(images)
                class_loss = class_criterion(class_scores, labels[:, 0].long())
                bbox_loss = bbox_criterion(bbox_outputs, labels[:, 1:])
                loss = class_loss + bbox_loss
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

    metadata_optimizer = str(optimizer).replace('\n', '')
    metadata_criterion = str(class_criterion).replace('\n', '') + ', ' + str(bbox_criterion).replace('\n', '')
    metadata_model = str(model).replace('\n', '')
    metadata = (f"epochs#{epoch+1}",
                f"batch#{BATCH_SIZE}",
                f"classes#{NUM_CLASSES}",
                f"bboxes#{NUM_BBOX}",
                f"outputs#{NUM_CLASSES+NUM_BBOX}",
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
                f"transform#{transform}",
                f"optimizer#{metadata_optimizer}",
                f"loss_function#{metadata_criterion}",
                f"training_size#{train_size}",
                f"validation_size#{val_size}",
                f"training_loss#{best_model_training_loss}",
                f"validation_loss#{best_model_validation_loss}")
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

    metadata_optimizer = str(optimizer).replace('\n', '')
    metadata_criterion = str(class_criterion).replace('\n', '') + ', ' + str(bbox_criterion).replace('\n', '')
    metadata_model = str(best_model).replace('\n', '')
    metadata = (f"epochs#{best_model_epoch+1}",
                f"batch#{BATCH_SIZE}",
                f"classes#{NUM_CLASSES}",
                f"bboxes#{NUM_BBOX}",
                f"outputs#{NUM_CLASSES+NUM_BBOX}",
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
                f"transform#{transform}",
                f"optimizer#{metadata_optimizer}",
                f"loss_function#{metadata_criterion}",
                f"training_size#{train_size}",
                f"validation_size#{val_size}",
                f"training_loss#{training_loss}",
                f"validation_loss#{validation_loss}")
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