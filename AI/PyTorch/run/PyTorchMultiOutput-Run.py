from torchvision import transforms
import vgamepad as vg
from PIL import Image
import torch.nn as nn
import numpy as np
import bettercam
import torch
import cv2
import os

IMG_WIDTH = 420
IMG_HEIGHT = 220

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = ""
for file in os.listdir(SCRIPT_PATH):
    if file.endswith(".pt"):
        MODEL_PATH = os.path.join(SCRIPT_PATH, file)
        break

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

camera = bettercam.create(output_color="BGR", output_idx=0)
gamepad = vg.VX360Gamepad()
lower_red = np.array([0, 0, 160])
upper_red = np.array([110, 110, 255])

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3)
        self.fc = nn.Linear(16*218*418, 3)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
model = Model().to(device)
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

while True:
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

    steering = output[0][1].item() * -1

    gamepad.left_joystick_float(x_value_float=steering, y_value_float=0)
    gamepad.update()
    
    predicted_values = output.squeeze().cpu().numpy()
    print(f"{predicted_values}, {steering}")