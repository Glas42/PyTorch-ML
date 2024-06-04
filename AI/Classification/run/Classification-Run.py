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

print(f"\nModel: {MODEL_PATH}")

metadata = {"data": []}
model = torch.jit.load(os.path.join(MODEL_PATH), _extra_files=metadata, map_location=device)
model.eval()
print(metadata)

transform = transforms.Compose([
    transforms.ToTensor(),
])

CLASSES = 1

total = len(os.listdir(f"{os.path.dirname(PATH)}\\EditedTrainingData")) // 2
correct = 0
incorrect = 0
counts = [0] * CLASSES
confidences = [0] * CLASSES
highest = [0] * CLASSES
lowest = [1] * CLASSES

for file in os.listdir(f"{os.path.dirname(PATH)}\\EditedTrainingData"):
    if file.endswith(".png"):

        frame = cv2.imread(os.path.join(f"{os.path.dirname(PATH)}\\EditedTrainingData", file))
        frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = transform(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            output = np.array(model(frame)[0].tolist())

        probabilities = np.exp(output - np.max(output)) / np.sum(np.exp(output - np.max(output)))
        obj_confidence = np.max(probabilities)
        obj_class = np.argmax(probabilities)

        counts[obj_class] += 1
        confidences[obj_class] += obj_confidence
        if obj_confidence > highest[obj_class]:
            highest[obj_class] = obj_confidence
        if obj_confidence < lowest[obj_class]:
            lowest[obj_class] = obj_confidence

        with open(os.path.join(f"{os.path.dirname(PATH)}\\EditedTrainingData", file.replace(".png", ".txt")), 'r') as f:
            content = f.read()
            if int(obj_class) == int(content):
                correct += 1
            else:
                incorrect += 1

for i in range(len(confidences)):
    confidence = confidences[i]
    print(f"Avg confidence of class {i}: {confidence / counts[i]}")
for i in range(len(highest)):
    print(f"Highest confidence of class {i}: {highest[i]}")
for i in range(len(lowest)):
    print(f"Lowest confidence of class {i}: {lowest[i]}")
print(f"Correct: {correct}\nIncorrect: {incorrect}")
print(f"Accuracy: {correct/total*100}%")