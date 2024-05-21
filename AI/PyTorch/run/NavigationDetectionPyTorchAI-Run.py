from torchvision import transforms
import torch.nn.functional as F
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

PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) + "\\ModelFiles\\Models"
MODEL_PATH = ""
for file in os.listdir(PATH):
    if file.endswith(".pt"):
        MODEL_PATH = os.path.join(PATH, file)
        break
IMG_WIDTH = 420
IMG_HEIGHT = 220

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 55 * 105, 500)
        self.fc2 = nn.Linear(500, 3)  # Assuming OUTPUTS = 3

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 55 * 105)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net().to(device)
model.load_state_dict(torch.load(os.path.join(MODEL_PATH), map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('frame', cv2.WND_PROP_TOPMOST, 1)
cv2.resizeWindow('frame', IMG_WIDTH, IMG_HEIGHT)

while True:
    start = time.time()
    frame = camera.grab()
    if frame is None:
        continue
    frame = frame[759:979, 1479:1899]
    cv2.rectangle(frame, (0,0), (round(frame.shape[1]/6),round(frame.shape[0]/3)),(0,0,0),-1)
    cv2.rectangle(frame, (frame.shape[1],0), (round(frame.shape[1]-frame.shape[1]/6),round(frame.shape[0]/3)),(0,0,0),-1)
    frame = cv2.inRange(frame, lower_red, upper_red)
    
    frame = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    frame = np.array(frame, dtype=np.float32) / 255.0
    
    with torch.no_grad():
        output = model(transform(frame).unsqueeze(0).to(device))
        output = output.tolist()
        print(output[0])

    steering = output[0][0] / -30
    left_indicator = output[0][1]
    right_indicator = output[0][2]

    gamepad.left_joystick_float(x_value_float=steering, y_value_float=0)
    gamepad.update()

    cv2.putText(frame, f"FPS: {round(1 / (time.time() - start), 1)}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Steering: {round(steering, 2)}", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Left Indicator: {round(left_indicator, 2)}", (5, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Right Indicator: {round(right_indicator, 2)}", (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()