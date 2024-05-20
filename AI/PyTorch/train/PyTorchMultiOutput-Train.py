from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import torch.nn as nn
import torch
import time
import os

EPOCHS = 100
BATCH_SIZE = 200
IMG_WIDTH = 420
IMG_HEIGHT = 220
OUTPUTS = 3

PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_PATH = PATH + "\\ModelFiles\\EditedTrainingData"
MODEL_PATH = PATH + "\\ModelFiles\\Models"

print("Training on " + "CUDA" if torch.cuda.is_available() else "CPU")

print("Loading...")

class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = [file for file in os.listdir(data_dir) if file.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.data_dir, image_name)
        image = Image.open(image_path).convert('L')
        txt_path = os.path.join(self.data_dir, image_name.replace('.png', '.txt'))
        with open(txt_path, 'r') as file:
            label = [float(x) for x in file.readline().split(',')]
        
        transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.Lambda(lambda x: x.convert("L")),
            transforms.Lambda(lambda x: x.point(lambda p: p > 128 and 255)),
            transforms.ToTensor()
        ])
        image = transform(image)
        
        return image, torch.tensor(label, dtype=torch.float)

dataset = CustomDataset(DATAPATH)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(1, OUTPUTS, kernel_size=3)
        self.fc1 = nn.Linear(OUTPUTS*218*418, 64)
        self.fc2 = nn.Linear(64, OUTPUTS)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model()
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

start_time = time.time()
update_time = start_time
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        print(f"\rEpoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}                                ", end='\r', flush=True)
        
    print(f"Epoch [{epoch+1}/{EPOCHS}], Average Loss: {running_loss / len(dataloader):.4f}, {round((time.time() - update_time) if time.time() - update_time > 1 else (time.time() - update_time) * 1000, 2)}{'s' if time.time() - update_time > 1 else 'ms'}/Epoch, ETA: {time.strftime('%H:%M:%S', time.gmtime(round((time.time() - start_time) / (epoch + 1) * EPOCHS - (time.time() - start_time), 2)))}                                ")
    update_time = time.time()

print("Training completed, saving model...")

torch.save(model.state_dict(), os.path.join(MODELPATH, f"EP-{EPOCHS}_BS-{BATCH_SIZE}_RES-{IMG_WIDTH}x{IMG_HEIGHT}_IMAGES-{len(dataset)}.pt"))

print("Model saved successfully, closing script...")