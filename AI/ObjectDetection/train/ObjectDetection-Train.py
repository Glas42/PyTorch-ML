import datetime
print(f"\n------------------------------------\n\n\033[90m[{datetime.datetime.now().strftime('%H:%M:%S')}] \033[0mImporting libraries...")

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
import multiprocessing
import torch.nn as nn
from PIL import Image
import numpy as np
import shutil
import torch
import time
import cv2

# Constants
PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_PATH = PATH + "\\ModelFiles\\EditedTrainingData"
MODEL_PATH = PATH + "\\ModelFiles\\Models"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_WIDTH = 420
IMG_HEIGHT = 220
NUM_EPOCHS = 200
BATCH_SIZE = 200
OUTPUTS = 5
DROPOUT = 0.5
LEARNING_RATE = 0.0001
TRAIN_VAL_RATIO = 0.8
NUM_WORKERS = 0
SHUFFLE = True
USE_FP16 = False
PATIENCE = 10
PIN_MEMORY = True

IMG_COUNT = 0
for file in os.listdir(DATA_PATH):
    if file.endswith(".png"):
        IMG_COUNT += 1

RED = "\033[91m"
GREEN = "\033[92m"
DARK_GREY = "\033[90m"
NORMAL = "\033[0m"
def timestamp():
    return DARK_GREY + f"[{datetime.datetime.now().strftime('%H:%M:%S')}] " + NORMAL

print("\n------------------------------------\n")

print(timestamp() + f"Using {str(DEVICE).upper()} for training")
print(timestamp() + 'Number of CPU cores:', multiprocessing.cpu_count())
print()
print(timestamp() + "Training settings:")
print(timestamp() + "> Epochs:", NUM_EPOCHS)
print(timestamp() + "> Batch size:", BATCH_SIZE)
print(timestamp() + "> Output size:", OUTPUTS)
print(timestamp() + "> Dropout:", DROPOUT)
print(timestamp() + "> Dataset split:", TRAIN_VAL_RATIO)
print(timestamp() + "> Learning rate:", LEARNING_RATE)
print(timestamp() + "> Number of workers:", NUM_WORKERS)
print(timestamp() + "> Shuffle:", SHUFFLE)
print(timestamp() + "> Use FP16:", USE_FP16)
print(timestamp() + "> Patience:", PATIENCE)
print(timestamp() + "> Pin memory:", PIN_MEMORY)
print(timestamp() + "> Image width:", IMG_WIDTH)
print(timestamp() + "> Image height:", IMG_HEIGHT)
print(timestamp() + "> Images:", IMG_COUNT)

print("\n------------------------------------\n")

print(timestamp() + "Loading...")

def load_data():
    images = []
    targets = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".png"):
            img = Image.open(os.path.join(DATA_PATH, file)).convert('L')  # Convert to grayscale
            img = np.array(img)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            target_file = os.path.join(DATA_PATH, file.replace(".png", ".txt"))
            if os.path.exists(target_file):
                with open(target_file, 'r') as f:
                    content = str(f.read()).split(',')
                    obj_x1 = float(content[0])
                    obj_y1 = float(content[1])
                    obj_x2 = float(content[2])
                    obj_y2 = float(content[3])
                    obj_class = int(0 if str(content[4]) == 'Green' else 1 if str(content[4]) == 'Yellow' else 2 if str(content[4]) == 'Red' else 2)
                    obj_present = 1.0 if obj_class != 2 else 0.0  # 1.0 if object is present, 0.0 if not present
                    target = [obj_x1, obj_y1, obj_x2, obj_y2, obj_present, obj_class]
                images.append(img)
                targets.append(target)
            else:
                pass

    return np.array(images, dtype=np.float16 if USE_FP16 else np.float32), np.array(targets, dtype=np.float16 if USE_FP16 else np.float32)  # Convert to float16 or float32

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, images, targets, transform=None):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float16 if USE_FP16 else torch.float32).unsqueeze(0)
            print(timestamp() + "Warning: No transformation applied to image.")
        return image, torch.tensor(target, dtype=torch.float16 if USE_FP16 else torch.float32)

# Define the model
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # Input channels = 1 for grayscale images
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self._to_linear = 64 * 52 * 27
        self.fc1 = nn.Linear(self._to_linear, 500)
        self.fc2 = nn.Linear(500, 6)  # Output size = 6 (4 bounding box coords + 1 object + 1 class)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 420x220 -> 210x110
        x = self.pool(F.relu(self.conv2(x)))  # 210x110 -> 105x55
        x = self.pool(F.relu(self.conv3(x)))  # 105x55 -> 52x27
        x = x.view(-1, self._to_linear)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class WeightedMSELoss(nn.Module):
    def __init__(self, bbox_weight=1.0, obj_weight=1.0, class_weight=1.0):
        super(WeightedMSELoss, self).__init__()
        self.bbox_weight = bbox_weight
        self.obj_weight = obj_weight
        self.class_weight = class_weight

    def forward(self, outputs, targets):
        if outputs.dim() == 1:
            bbox_loss = F.mse_loss(outputs[:4], targets[:4], reduction='mean')
            obj_loss = F.mse_loss(outputs[4], targets[4], reduction='mean')
            class_loss = F.mse_loss(outputs[5], targets[5], reduction='mean')
        else:
            bbox_loss = F.mse_loss(outputs[:, :4], targets[:, :4], reduction='mean')
            obj_loss = F.mse_loss(outputs[:, 4], targets[:, 4], reduction='mean')
            class_loss = F.mse_loss(outputs[:, 5], targets[:, 5], reduction='mean')
        return self.bbox_weight * bbox_loss, self.obj_weight * obj_loss, self.class_weight * class_loss

def main():
    # Load data
    images, targets = load_data()

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create dataset
    dataset = CustomDataset(images, targets, transform=transform)

    # Initialize model, loss function, and optimizer
    model = ConvolutionalNeuralNetwork().to(DEVICE)
    if USE_FP16:
        model = model.half()  # Convert the model to use 16-bit float format
    criterion = WeightedMSELoss(bbox_weight=100.0, obj_weight=1.0, class_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Split the dataset into training and validation sets
    train_size = int(TRAIN_VAL_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # Early stopping variables
    best_val_loss = float('inf')
    best_model = None
    best_model_epoch = None
    wait = 0

    # Create tensorboard logs folder if it doesn't exist
    if not os.path.exists(f"{PATH}/AI/ObjectDetection/logs"):
        os.makedirs(f"{PATH}/AI/ObjectDetection/logs")

    # Delete previous tensorboard logs
    for obj in os.listdir(f"{PATH}/AI/ObjectDetection/logs"):
        try:
            shutil.rmtree(f"{PATH}/AI/ObjectDetection/logs/{obj}")
        except:
            os.remove(f"{PATH}/AI/ObjectDetection/logs/{obj}")

    # Tensorboard setup
    summary_writer = SummaryWriter(f"{PATH}/AI/ObjectDetection/logs", comment="ObjectDetection-Training", flush_secs=20)

    print(timestamp() + "Starting training...")
    print("\n------------------------------------------------------------------------------------------------------\n")
    start_time = time.time()
    update_time = start_time

    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        running_bbox_loss = 0.0
        running_obj_loss = 0.0
        running_class_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)
            bbox_loss, obj_loss, class_loss = criterion(outputs, labels)
            loss = bbox_loss + obj_loss + class_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_bbox_loss += bbox_loss.item()
            running_obj_loss += obj_loss.item()
            running_class_loss += class_loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_bbox_loss = 0.0
        val_obj_loss = 0.0
        val_class_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_dataloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                outputs = model(inputs)
                bbox_loss, obj_loss, class_loss = criterion(outputs, labels)
                val_loss += (bbox_loss + obj_loss + class_loss).item()
                val_bbox_loss += bbox_loss.item()
                val_obj_loss += obj_loss.item()
                val_class_loss += class_loss.item()

        val_loss /= len(val_dataloader)
        val_bbox_loss /= len(val_dataloader)
        val_obj_loss /= len(val_dataloader)
        val_class_loss /= len(val_dataloader)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            best_model_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"\rEarly stopping at Epoch {epoch+1}, Train Loss: {running_loss / len(train_dataloader)}, Val Loss: {val_loss}                       ", end='', flush=True)
                break

        # Log values to Tensorboard
        summary_writer.add_scalars(f'Loss', {
            'train': running_loss / len(train_dataloader),
            'validation': val_loss,
            'train_bbox': running_bbox_loss / len(train_dataloader),
            'train_obj': running_obj_loss / len(train_dataloader),
            'train_class': running_class_loss / len(train_dataloader),
            'val_bbox': val_bbox_loss,
            'val_obj': val_obj_loss,
            'val_class': val_class_loss,
        }, epoch)

        print(f"\rEpoch {epoch+1}, Train Loss: {running_loss / len(train_dataloader)}, Val Loss: {val_loss}, {round((time.time() - update_time) if time.time() - update_time > 1 else (time.time() - update_time) * 1000, 2)}{'s' if time.time() - update_time > 1 else 'ms'}/Epoch, ETA: {time.strftime('%H:%M:%S', time.gmtime(round((time.time() - start_time) / (epoch + 1) * NUM_EPOCHS - (time.time() - start_time), 2)))}                       ", end='', flush=True)
        update_time = time.time()

    print("\n\n------------------------------------------------------------------------------------------------------")

    TRAINING_TIME = time.strftime('%H-%M-%S', time.gmtime(time.time() - start_time))
    TRAINING_DATE = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    print()
    print(timestamp() + f"Training completed after " + TRAINING_TIME.replace('-', ':'))

    # Save the last model
    print(timestamp() + "Saving the last model...")
    last_model_saved = False
    for i in range(5):
        try:
            last_model = torch.jit.script(model)
            torch.jit.save(last_model, os.path.join(MODEL_PATH, f"ObjectDetectionModel-LAST_EPOCHS-{epoch+1}_BATCH-{BATCH_SIZE}_IMG_WIDTH-{IMG_WIDTH}_IMG_HEIGHT-{IMG_HEIGHT}_IMG_COUNT-{IMG_COUNT}_TIME-{TRAINING_TIME}_DATE-{TRAINING_DATE}.pt"))
            last_model_saved = True
            break
        except:
            print(timestamp() + "Failed to save the last model. Retrying...")
    print(timestamp() + "Last model saved successfully.") if last_model_saved else print(timestamp() + "Failed to save the last model.")

    # Save the best model
    print(timestamp() + "Saving the best model...")
    best_model_saved = False
    for i in range(5):
        try:
            best_model = torch.jit.script(best_model)
            torch.jit.save(best_model, os.path.join(MODEL_PATH, f"ObjectDetectionModel-BEST_EPOCHS-{best_model_epoch+1}_BATCH-{BATCH_SIZE}_IMG_WIDTH-{IMG_WIDTH}_IMG_HEIGHT-{IMG_HEIGHT}_IMG_COUNT-{IMG_COUNT}_TIME-{TRAINING_TIME}_DATE-{TRAINING_DATE}.pt"))
            best_model_saved = True
            break
        except:
            print(timestamp() + "Failed to save the best model. Retrying...")
    print(timestamp() + "Best model saved successfully.") if best_model_saved else print(timestamp() + "Failed to save the best model.")

    print("\n------------------------------------\n")

if __name__ == '__main__':
    main()