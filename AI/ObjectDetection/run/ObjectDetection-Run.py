from torchvision import transforms
import numpy as np
import bettercam
import torch
import time
import cv2
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
camera = bettercam.create(output_color="BGR", output_idx=0)

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
OUTPUTS = 5

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

    frame = np.array(frame)
    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    with torch.no_grad():
        output = model(transform(frame).unsqueeze(0).to(device))

    obj_x1, obj_y1, obj_x2, obj_y2, obj_class = output[0].tolist()
    cv2.rectangle(frame, (int(obj_x1 * frame.shape[1]), int(obj_y1 * frame.shape[0])), (int(obj_x2 * frame.shape[1]), int(obj_y2 * frame.shape[0])), (255, 255, 255), 2)

    cv2.putText(frame, f"FPS: {round(1 / (time.time() - start), 1)}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()