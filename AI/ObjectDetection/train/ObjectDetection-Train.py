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
DATA_PATH = PATH + "\ModelFiles\EditedTrainingData"
MODEL_PATH = PATH + "\ModelFiles\Models"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_WIDTH = 640
IMG_HEIGHT = 640
NUM_EPOCHS = 200
BATCH_SIZE = 10
NUM_CLASSES = 3
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
            img = np.array(img, dtype=np.float16 if USE_FP16 else np.float32) / 255.0  # Convert to float16 or float32

            target_file = os.path.join(DATA_PATH, file.replace(".png", ".txt"))
            if os.path.exists(target_file):
                with open(target_file, 'r') as f:
                    content = str(f.read()).split(',')
                    obj_x1 = float(content[0])
                    obj_y1 = float(content[1])
                    obj_x2 = float(content[2])
                    obj_y2 = float(content[3])
                    obj_class = int(0 if str(content[4]) == 'Green' else 1 if str(content[4]) == 'Yellow' else 2 if str(content[4]) == 'Red' else 2)
                    target = [obj_x1, obj_y1, obj_x2, obj_y2, obj_class]
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
        target = torch.tensor(target, dtype=torch.float16 if USE_FP16 else torch.float32)
        return image, target

# SSD model components
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class SSD(nn.Module):
    def __init__(self, num_classes):
        super(SSD, self).__init__()
        self.num_classes = num_classes

        self.conv1 = BasicConv(1, 64, kernel_size=3, stride=1, padding=1)  # Adjusted for single channel input
        self.conv2 = BasicConv(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = BasicConv(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = BasicConv(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv5 = BasicConv(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv6 = BasicConv(512, 256, kernel_size=3, stride=2, padding=1)

        self.loc = nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)
        self.conf = nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        loc = self.loc(x).permute(0, 2, 3, 1).contiguous()
        conf = self.conf(x).permute(0, 2, 3, 1).contiguous()

        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)
        
        return loc, conf

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    
    loc_targets = []
    conf_targets = []
    for target in targets:
        loc_target = torch.zeros(1600, 4)  # Create a tensor of zeros with shape [1600, 4]
        loc_target[:target.shape[0], :] = target[:4]  # Copy the values from target to loc_target
        loc_targets.append(loc_target)
        conf_target = torch.zeros(1600, dtype=torch.long)  # Create a tensor of zeros with shape [1600]
        conf_target[:target.shape[0]] = target[4]  # Copy the values from target to conf_target
        conf_targets.append(conf_target)

    loc_targets = torch.stack(loc_targets, dim=0)
    conf_targets = torch.stack(conf_targets, dim=0)
    
    return images, (loc_targets, conf_targets)

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
    model = SSD(num_classes=NUM_CLASSES).to(DEVICE)
    if USE_FP16:
        model = model.half()

    loc_criterion = nn.SmoothL1Loss()
    conf_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Split the dataset into training and validation sets
    train_size = int(TRAIN_VAL_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=collate_fn)

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

    # Training and Validation Loops
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs, (loc_labels, conf_labels) = inputs.to(DEVICE), (labels[0].to(DEVICE), labels[1].to(DEVICE))

            optimizer.zero_grad()

            outputs = model(inputs)
            loc_loss = loc_criterion(outputs[0], loc_labels)
            conf_loss = conf_criterion(outputs[1].view(-1, NUM_CLASSES), conf_labels.view(-1))
            loss = loc_loss + conf_loss
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_dataloader, 0):
                inputs, labels = data
                inputs, (loc_labels, conf_labels) = inputs.to(DEVICE), (labels[0].to(DEVICE), labels[1].to(DEVICE))

                outputs = model(inputs)
                loc_loss = loc_criterion(outputs[0], loc_labels)
                conf_loss = conf_criterion(outputs[1].view(-1, NUM_CLASSES), conf_labels.view(-1))
                loss = loc_loss + conf_loss
                val_loss += loss.item()

        val_loss /= len(val_dataloader)

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