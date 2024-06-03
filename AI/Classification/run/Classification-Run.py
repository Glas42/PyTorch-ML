from torchvision import transforms
import numpy as np
import torch
import cv2
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) + "\\ModelFiles\\Models"
MODEL_PATH = ""
for file in os.listdir(PATH):
    if file.endswith(".pt"):
        MODEL_PATH = os.path.join(PATH, file)
        break
if MODEL_PATH == "":
    print("No model found.")
    exit()

IMG_WIDTH = 80
IMG_HEIGHT = 160

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

total = len(os.listdir(f"{os.path.dirname(PATH)}\\EditedTrainingData")) // 2
correct = 0

for file in os.listdir(f"{os.path.dirname(PATH)}\\EditedTrainingData"):
    if file.endswith(".png"):

        frame = cv2.imread(os.path.join(f"{os.path.dirname(PATH)}\\EditedTrainingData", file))
        frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        with torch.no_grad():
            output = model(transform(frame).unsqueeze(0).to(device))

        obj_class = output[0].tolist()
        obj_class = np.argmax(obj_class)
        with open(os.path.join(f"{os.path.dirname(PATH)}\\EditedTrainingData", file.replace(".png", ".txt")), 'r') as f:
            content = f.read()
            print(f"{int(obj_class) == int(content)} {int(obj_class)} {int(content)}")
            if int(obj_class) == int(content):
                correct += 1

print(f"Accuracy: {correct/total*100}%")