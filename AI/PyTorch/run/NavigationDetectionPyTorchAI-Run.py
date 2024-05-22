from SDKController import SCSController
from TruckSimAPI import scsTelemetry
from torchvision import transforms
import numpy as np
import bettercam
import torch
import time
import cv2
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
camera = bettercam.create(output_color="BGR", output_idx=0)
controller = SCSController()

API = scsTelemetry()
data = API.update()
if data["scsValues"]["telemetryPluginRevision"] < 2:
    print("TruckSimAPI is waiting for the game...")
while data["scsValues"]["telemetryPluginRevision"] < 2:
    time.sleep(0.1)
    data = API.update()

lower_red = np.array([0, 0, 160])
upper_red = np.array([110, 110, 255])

PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) + "\\ModelFiles\\Models"
MODEL_PATH = ""
for file in os.listdir(PATH):
    if file.endswith(".pt"):
        MODEL_PATH = os.path.join(PATH, file)
        break
if MODEL_PATH == "":
    print("No model found.")
    exit()

IMG_WIDTH = 420
IMG_HEIGHT = 220
OUTPUTS = 3

string = MODEL_PATH.split("\\")[-1]
epochs = int(string.split("EPOCHS-")[1].split("_")[0])
batch = int(string.split("BATCH-")[1].split("_")[0])
img_width = int(string.split("IMG_WIDTH-")[1].split("_")[0])
img_height = int(string.split("IMG_HEIGHT-")[1].split("_")[0])
img_count = int(string.split("IMG_COUNT-")[1].split("_")[0])
training_time = string.split("TIME-")[1].split("_")[0]
training_date = string.split("DATE-")[1].split(".")[0]

print(f"\nModel: {MODEL_PATH}")
print(f"\n> Epochs: {epochs}")
print(f"> Batch: {batch}")
print(f"> Image Width: {img_width}")
print(f"> Image Height: {img_height}")
print(f"> Image Count: {img_count}")
print(f"> Training Time: {training_time}")
print(f"> Training Date: {training_date}\n")

model = torch.jit.load(os.path.join(MODEL_PATH), map_location=device)
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
    cv2.rectangle(frame, (0, 0), (round(frame.shape[1]/6), round(frame.shape[0]/3)), (0, 0, 0), -1)
    cv2.rectangle(frame, (frame.shape[1], 0), (round(frame.shape[1]-frame.shape[1]/6), round(frame.shape[0]/3)), (0, 0, 0), -1)
    frame = cv2.inRange(frame, lower_red, upper_red)
    frame = np.array(frame)
    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    frame = np.array(frame, dtype=np.float32) / 255.0

    data["api"] = API.update()
    if data["scsValues"]["telemetryPluginRevision"] < 2:
        print("TruckSimAPI is waiting for the game...")
        continue

    with torch.no_grad():
        output = model(transform(frame).unsqueeze(0).to(device)) # , torch.tensor(data["api"]["truckFloat"]["speedLimit"]).unsqueeze(0).to(device)
        output = output.tolist()

    steering = float(output[0][0] / -30)
    left_indicator = float(output[0][1])
    right_indicator = float(output[0][2])
    left_indicator_bool = bool(left_indicator > 0.3)
    right_indicator_bool = bool(right_indicator > 0.3)
    #speed = float(output[0][3])

    controller.steering = steering
    controller.lblinker = left_indicator_bool
    controller.rblinker = right_indicator_bool

    cv2.putText(frame, f"FPS: {round(1 / (time.time() - start), 1)}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Steer: {round(steering, 2)}", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Left: {round(left_indicator, 2)}", (5, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Right: {round(right_indicator, 2)}", (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    #cv2.putText(frame, f"Speed: {round(speed*3.6, 2)}", (5, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
controller.close()