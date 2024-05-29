from torchvision import transforms
import numpy as np
import bettercam
import torch
import time
import cv2
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
camera = bettercam.create(output_color="BGR", output_idx=0)

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
    width = frame.shape[1]
    height = frame.shape[0]
    frame = np.array(frame)
    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    with torch.no_grad():
        output = model(transform(frame).unsqueeze(0).to(device))
        output = output.tolist()

    obj_x1, obj_y1, obj_x2, obj_y2, obj_class = output[0][0], output[0][1], output[0][2], output[0][3], "Green" if int(output[0][4]) == 1 else ("Yellow" if int(output[0][4]) == 2 else "Red")

    print(f"X1: {obj_x1 * width}, Y1: {obj_y1 * height}, X2: {obj_x2 * width}, Y2: {obj_y2 * height}, Class: {obj_class}")

    cv2.rectangle(frame, (int(obj_x1 * width), int(obj_y1 * height)), (int(obj_x2 * width), int(obj_y2 * height)), (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, f"FPS: {round(1 / (time.time() - start), 1)}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()