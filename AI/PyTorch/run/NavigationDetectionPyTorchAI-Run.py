from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import vgamepad as vg
import numpy as np
import bettercam
import torch
import time
import cv2
import os

# Set device to CUDA if available, otherwise fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

camera = bettercam.create(output_color="BGR", output_idx=0)
gamepad = vg.VX360Gamepad()
lower_red = np.array([0, 0, 160])
upper_red = np.array([110, 110, 255])

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = ""
for file in os.listdir(SCRIPT_PATH):
    if file.endswith(".pt"):
        MODEL_PATH = os.path.join(SCRIPT_PATH, file)
        break
IMG_WIDTH = 420
IMG_HEIGHT = 220

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Adjust input channels to 1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 27 * 52, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 27 * 52)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net().to(device)
model.load_state_dict(torch.load(os.path.join(MODEL_PATH), map_location=device))
model.eval()

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy array to PIL image
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.Grayscale(),
        transforms.Lambda(lambda x: x.point(lambda p: p > 128 and 255)),
        transforms.ToTensor()
    ])
    image_pil = transform(image)
    return image_pil.unsqueeze(0).to(device)

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('frame', cv2.WND_PROP_TOPMOST, 1)
cv2.resizeWindow('frame', IMG_WIDTH, IMG_HEIGHT)

while True:
    start = time.time()
    frame = camera.grab()
    if type(frame) == type(None):
        continue
    frame = frame[759:979, 1479:1899]
    cv2.rectangle(frame, (0,0), (round(frame.shape[1]/6),round(frame.shape[0]/3)),(0,0,0),-1)
    cv2.rectangle(frame, (frame.shape[1],0), (round(frame.shape[1]-frame.shape[1]/6),round(frame.shape[0]/3)),(0,0,0),-1)
    frame = cv2.inRange(frame, lower_red, upper_red)
    frame = preprocess_image(frame)
    with torch.no_grad():
        output = model(frame)
        output = output.tolist()
        print(output)

    steering = output[0][1] * -100

    gamepad.left_joystick_float(x_value_float=steering, y_value_float=0)
    gamepad.update()

    frame = frame.cpu().numpy().squeeze() * 255
    frame = frame.astype(np.uint8)

    cv2.putText(frame, f"FPS: {round(1 / (time.time() - start), 1)}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break