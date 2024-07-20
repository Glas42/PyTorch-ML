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

print(f"\nModel: {MODEL_PATH}")

metadata = {"data": []}
model = torch.jit.load(os.path.join(MODEL_PATH), _extra_files=metadata, map_location=device)
model.eval()

metadata = str(metadata["data"]).replace('b"(', '').replace(')"', '').replace("'", "").split(", ") # now in the format: ["key#value", "key#value", ...]
for var in metadata:
    if "classes" in var:
        CLASSES = int(var.split("#")[1])
    if "image_width" in var:
        IMG_WIDTH = int(var.split("#")[1])
    if "image_height" in var:
        IMG_HEIGHT = int(var.split("#")[1])
    if "image_channels" in var:
        IMG_CHANNELS = str(var.split("#")[1])
    if "training_dataset_accuracy" in var:
        print("Training dataset accuracy: " + str(var.split("#")[1]))
    if "validation_dataset_accuracy" in var:
        print("Validation dataset accuracy: " + str(var.split("#")[1]))
    if "val_transform" in var:
        transform = var.replace("\\n", "\n").replace('\\', '').split("#")[1]
        transform_list = []
        transform_parts = transform.strip().split("\n")
        for part in transform_parts[1:-1]:
            part = part.strip()
            if part:
                try:
                    transform_args = []
                    transform_name = part.split("(")[0]
                    if "(" in part:
                        args = part.split("(")[1][:-1].split(",")
                        for arg in args:
                            try:
                                transform_args.append(int(arg.strip()))
                            except ValueError:
                                try:
                                    transform_args.append(float(arg.strip()))
                                except ValueError:
                                    transform_args.append(arg.strip())
                    if transform_name == "ToTensor":
                        transform_list.append(transforms.ToTensor())
                    else:
                        transform_list.append(getattr(transforms, transform_name)(*transform_args))
                except (AttributeError, IndexError, ValueError):
                    print(f"Skipping or failed to create transform: {part}")
        transform = transforms.Compose(transform_list)

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
        frame = np.array(frame, dtype=np.float32)
        if IMG_CHANNELS == 'Grayscale' or IMG_CHANNELS == 'Binarize':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if IMG_CHANNELS == 'RG':
            frame = np.stack((frame[:, :, 0], frame[:, :, 1]), axis=2)
        elif IMG_CHANNELS == 'GB':
            frame = np.stack((frame[:, :, 1], frame[:, :, 2]), axis=2)
        elif IMG_CHANNELS == 'RB':
            frame = np.stack((frame[:, :, 0], frame[:, :, 2]), axis=2)
        elif IMG_CHANNELS == 'R':
            frame = frame[:, :, 0]
            frame = np.expand_dims(frame, axis=2)
        elif IMG_CHANNELS == 'G':
            frame = frame[:, :, 1]
            frame = np.expand_dims(frame, axis=2)
        elif IMG_CHANNELS == 'B':
            frame = frame[:, :, 2]
            frame = np.expand_dims(frame, axis=2)

        frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        frame = frame / 255.0

        if IMG_CHANNELS == 'Binarize':
            frame = cv2.threshold(frame, 0.5, 1.0, cv2.THRESH_BINARY)[1]


        frame = transform(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            output = np.array(model(frame)[0].tolist())

        output = output * (1 / sum(output))
        confidence = [x / sum(output) for x in output]
        obj_class = np.argmax(output)
        obj_confidence = confidence[obj_class]

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
    print(f"Avg confidence of class {i}: {(confidence / counts[i]) if counts[i] > 0 else 'no data'}")
for i in range(len(highest)):
    print(f"Highest confidence of class {i}: {highest[i]}")
for i in range(len(lowest)):
    print(f"Lowest confidence of class {i}: {lowest[i]}")
print(f"Correct: {correct}\nIncorrect: {incorrect}")
print(f"Accuracy: {correct/total*100}%")