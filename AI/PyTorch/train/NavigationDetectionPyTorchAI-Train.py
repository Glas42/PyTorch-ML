print("\n------------------------------------\n\nImporting libraries...")

from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import multiprocessing
import torch.nn as nn
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
IMG_HEIGHT = 220
IMG_WIDTH = 420
NUM_EPOCHS = 300
BATCH_SIZE = 200

print("\n------------------------------------\n")

print(f"CUDA available: {torch.cuda.is_available()}")

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} for training")

# Determine the number of CPU cores
num_cpu_cores = multiprocessing.cpu_count()
print('Number of CPU cores:', num_cpu_cores)

image_count = 0
for file in os.listdir(DATA_PATH):
    if file.endswith(".png"):
        image_count += 1

print("\nTraining settings:")
print("> Epochs:", NUM_EPOCHS)
print("> Batch size:", BATCH_SIZE)
print("> Image width:", IMG_WIDTH)
print("> Image height:", IMG_HEIGHT)
print("> Images:", image_count)

print("\n------------------------------------\n")

print("Loading...")

# Define custom dataset
class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.images, self.user_inputs = self.load_data(data_path)
    
    def load_data(self, data_path):
        images = []
        user_inputs = []
        for file in os.listdir(data_path):
            if file.endswith(".png"):
                # Load image
                img = Image.open(os.path.join(data_path, file))
                img = np.array(img)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                img_array = np.array(img) / 255.0
                
                # Load steering angle if corresponding file exists
                user_inputs_file = os.path.join(data_path, file.replace(".png", ".txt"))
                if os.path.exists(user_inputs_file):
                    with open(user_inputs_file, 'r') as f:
                        user_input = float(f.read().strip())
                    images.append(img_array)
                    user_inputs.append(user_input)
                else:
                    pass
        
        return np.array(images), np.array(user_inputs)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        user_input = self.user_inputs[idx]
        if self.transform:
            image = self.transform(image)
        return image, user_input

# Define transformation
transform = transforms.Compose([
    transforms.Lambda(lambda x: to_pil_image(x)),  # Convert to PIL Image
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.Lambda(lambda x: x.convert("L")),   # Convert to grayscale
    transforms.Lambda(lambda x: x.point(lambda p: p > 128 and 255)),  # Convert to binary
    transforms.ToTensor()
])

# Load data
dataset = CustomDataset(DATA_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Adjust input channels to 1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 27 * 52, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 27 * 52)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net().to(device)  # Move model to GPU if available

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

print("Starting training...")
print("\n--------------------------------------------------------------\n")
start_time = time.time()
update_time = start_time

# Train model
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # Explicitly convert inputs and labels to torch.float32
        inputs = inputs.float()
        labels = labels.float()
        optimizer.zero_grad()
        outputs = model(inputs)  # No need to call .float() here
        loss = criterion(outputs, labels.unsqueeze(1))
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