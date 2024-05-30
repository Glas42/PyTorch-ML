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
BATCH_SIZE = 32
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
                    obj_class = int(1 if str(content[4]) == 'Green' else 2 if str(content[4]) == 'Yellow' else 3 if str(content[4]) == 'Red' else 0)
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
        return image, torch.tensor(target, dtype=torch.float16 if USE_FP16 else torch.float32)

# Define the model
class Conv(nn.Module):
    # Standard convolution layer with BatchNorm and SiLU
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

def autopad(k, p=None):  # Pad to 'same'
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = Conv(1, 16, 3, 1)
        self.conv2 = Conv(16, 32, 3, 2)
        self.conv3 = Conv(32, 64, 3, 2)
        self.conv4 = Conv(64, 128, 3, 2)
        self.conv5 = Conv(128, 256, 3, 2)
        self.conv6 = Conv(256, 512, 3, 2)
        self.conv7 = Conv(512, 1024, 3, 2)
        
        # Update the input size of the fully connected layer based on the identified dimensions
        self._to_linear = 1024 * (IMG_WIDTH // 64) * (IMG_HEIGHT // 64)  # 1024 * 10 * 10 = 102400
        self.fc = nn.Linear(self._to_linear, OUTPUTS)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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
    criterion = nn.MSELoss()
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
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_dataloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
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