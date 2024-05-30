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
import traceback
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
BATCH_SIZE = 75
NUM_CLASSES = 3
NUM_BBOX = 1
BBOX_COORDS = 4
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
print(timestamp() + "> Number of classes:", NUM_CLASSES)
print(timestamp() + "> Number of bounding boxes per image:", NUM_BBOX)
print(timestamp() + "> Bounding box coordinates:", BBOX_COORDS)
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
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, num_bbox, bbox_coords, num_classes):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.num_bbox = num_bbox
        self.bbox_coords = bbox_coords
        self.num_classes = num_classes

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Region Proposal Network (RPN)
        self.rpn_conv = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.rpn_relu = nn.ReLU()
        self.rpn_cls_layer = nn.Conv2d(256, 2 * self.num_bbox, kernel_size=1)
        self.rpn_reg_layer = nn.Conv2d(256, 4 * self.num_bbox, kernel_size=1)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * (IMG_HEIGHT // 8) * (IMG_WIDTH // 8), 512)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(512, self.num_bbox * (self.bbox_coords + self.num_classes))

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Region Proposal Network
        rpn_feature = self.rpn_conv(x)
        rpn_feature = self.rpn_relu(rpn_feature)
        rpn_cls_output = self.rpn_cls_layer(rpn_feature)
        rpn_reg_output = self.rpn_reg_layer(rpn_feature)

        # Fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return rpn_cls_output, rpn_reg_output, x.view(x.size(0), self.num_bbox, self.bbox_coords + self.num_classes)

def custom_loss(rpn_cls_output, rpn_reg_output, detection_output, targets):
    # RPN classification loss
    rpn_cls_target = torch.zeros_like(rpn_cls_output)
    # Assign ground truth labels to the RPN classification targets
    rpn_cls_loss = F.binary_cross_entropy_with_logits(rpn_cls_output, rpn_cls_target)

    # RPN regression loss
    rpn_reg_target = torch.zeros_like(rpn_reg_output)
    # Assign ground truth bounding box coordinates to the RPN regression targets
    rpn_reg_loss = F.smooth_l1_loss(rpn_reg_output, rpn_reg_target)

    # Object detection loss
    detection_target = torch.zeros_like(detection_output)
    # Assign ground truth class labels and bounding box coordinates to the detection targets
    detection_loss = F.mse_loss(detection_output, detection_target)

    total_loss = rpn_cls_loss + rpn_reg_loss + detection_loss
    return total_loss

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
    model = ConvolutionalNeuralNetwork(NUM_BBOX, BBOX_COORDS, NUM_CLASSES).to(DEVICE)
    if USE_FP16:
        model = model.half()
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
        for batch_idx, (images, targets) in enumerate(train_dataloader):
            images = images.to(DEVICE)
            targets = [t.to(DEVICE) for t in targets]

            optimizer.zero_grad()

            rpn_cls_output, rpn_reg_output, detection_output = model(images)
            loss = custom_loss(rpn_cls_output, rpn_reg_output, detection_output, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_dataloader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_dataloader):
                images = images.to(DEVICE)
                targets = [t.to(DEVICE) for t in targets]

                rpn_cls_output, rpn_reg_output, detection_output = model(images)
                loss = custom_loss(rpn_cls_output, rpn_reg_output, detection_output, targets)
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
            print(timestamp() + f"Failed to save the last model: {traceback.format_exc()} - Retrying...")
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
            print(timestamp() + f"Failed to save the best model: {traceback.format_exc()} - Retrying...")
    print(timestamp() + "Best model saved successfully.") if best_model_saved else print(timestamp() + "Failed to save the best model.")

    print("\n------------------------------------\n")

if __name__ == '__main__':
    main()