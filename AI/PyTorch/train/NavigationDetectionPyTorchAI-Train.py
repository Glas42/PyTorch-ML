print("\n------------------------------------\n\nImporting libraries...")

from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import multiprocessing
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import datetime
import torch
import time
import cv2
import os

# Constants
PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_PATH = PATH + "\\ModelFiles\\EditedTrainingData"
MODEL_PATH = PATH + "\\ModelFiles\\Models"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_HEIGHT = 220
IMG_WIDTH = 420
NUM_EPOCHS = 50
BATCH_SIZE = 50
OUTPUTS = 3

image_count = 0
for file in os.listdir(DATA_PATH):
    if file.endswith(".png"):
        image_count += 1

print("\n------------------------------------\n")

print(f"Using {str(DEVICE).upper()} for training")
print('Number of CPU cores:', multiprocessing.cpu_count())

print("\nTraining settings:")
print("> Epochs:", NUM_EPOCHS)
print("> Batch size:", BATCH_SIZE)
print("> Output size:", OUTPUTS)
print("> Image width:", IMG_WIDTH)
print("> Image height:", IMG_HEIGHT)
print("> Images:", image_count)

print("\n------------------------------------\n")

print("Loading...")

def load_data(): 
    images = []
    user_inputs = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".png"):
            img = Image.open(os.path.join(DATA_PATH, file)).convert('L')  # Convert to grayscale
            img = np.array(img)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = np.array(img, dtype=np.float32) / 255.0  # Convert to float32

            user_inputs_file = os.path.join(DATA_PATH, file.replace(".png", ".txt"))
            if os.path.exists(user_inputs_file):
                with open(user_inputs_file, 'r') as f:
                    content = str(f.read()).split(',')
                    steering, left_indicator, right_indicator = float(content[0]), 1 if str(content[1]) == 'True' else 0, 1 if str(content[2]) == 'True' else 0
                    user_input = [steering, left_indicator, right_indicator]
                images.append(img)
                user_inputs.append(user_input)
            else:
                pass
    
    return np.array(images, dtype=np.float32), np.array(user_inputs, dtype=np.float32)  # Convert to float32

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
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
            print("Warning: No transformation applied to image.")
        return image, torch.tensor(user_input, dtype=torch.float32)

# Define the model
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # Input channels = 1 for grayscale images
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 27 * 52, 500)
        self.fc2 = nn.Linear(500, OUTPUTS)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 27 * 52)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    # Load data
    images, user_inputs = load_data()

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create dataset and dataloader
    dataset = CustomDataset(images, user_inputs, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize model, loss function, and optimizer
    model = ConvolutionalNeuralNetwork().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting training...")
    print("\n--------------------------------------------------------------\n")
    start_time = time.time()
    update_time = start_time

    # Train model
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"\rEpoch {epoch+1}, Loss: {running_loss / len(dataloader)}, {round((time.time() - update_time) if time.time() - update_time > 1 else (time.time() - update_time) * 1000, 2)}{'s' if time.time() - update_time > 1 else 'ms'}/Epoch, ETA: {time.strftime('%H:%M:%S', time.gmtime(round((time.time() - start_time) / (epoch + 1) * NUM_EPOCHS - (time.time() - start_time), 2)))}                       ", end='', flush=True)
        update_time = time.time()

    print("\n\n--------------------------------------------------------------")

    print("\nTraining completed in " + time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    # Save model
    print("Saving model...")
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, f"EPOCHS-{NUM_EPOCHS}_BATCH-{BATCH_SIZE}_RES-{IMG_WIDTH}x{IMG_HEIGHT}_IMAGES-{len(dataset)}_TRAININGTIME-{time.strftime('%H-%M-%S', time.gmtime(time.time() - start_time))}_DATE-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.pt"))
    print("Model saved successfully.")

    print("\n------------------------------------\n")

if __name__ == '__main__':
    main()