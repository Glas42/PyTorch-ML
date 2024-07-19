from torchvision import transforms
import numpy as np
import bettercam
import torch
import time
import cv2
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
camera = bettercam.create(output_color="BGR", output_idx=0)

PATH = "C:/GitHub/PyTorch-ML/ModelFiles/Models"
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
    if "transform" in var:
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

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('frame', cv2.WND_PROP_TOPMOST, 1)
cv2.resizeWindow('frame', round(IMG_WIDTH*2), round(IMG_HEIGHT))
cv2.namedWindow('left_mirror', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('left_mirror', cv2.WND_PROP_TOPMOST, 1)
cv2.resizeWindow('left_mirror', 300, 300)
cv2.namedWindow('right_mirror', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('right_mirror', cv2.WND_PROP_TOPMOST, 1)
cv2.resizeWindow('right_mirror', 300, 300)

background = np.zeros((IMG_HEIGHT, IMG_WIDTH*2, 3), dtype=np.uint8)
graph_background = np.zeros((300, 300, 3), dtype=np.uint8)

while True:
    start = time.time()

    frame_original = camera.grab()
    if frame_original is None:
        continue

    mirrorDistanceFromLeft = 23
    mirrorDistanceFromTop = 90
    mirrorWidth = 273
    mirrorHeight = 362
    scale = 1

    xCoord = (mirrorDistanceFromLeft * scale)
    yCoord = (mirrorDistanceFromTop * scale)
    left_top_left = (round(xCoord), round(yCoord))

    xCoord = (mirrorDistanceFromLeft * scale + mirrorWidth * scale)
    yCoord = (mirrorDistanceFromTop * scale + mirrorHeight * scale)
    left_bottom_right = (round(xCoord), round(yCoord))

    right_top_left = 1920 - left_bottom_right[0] - 1, left_top_left[1]
    right_bottom_right = 1920 - left_top_left[0] - 1, left_bottom_right[1]

    coords = [(left_top_left, left_bottom_right), (right_top_left, right_bottom_right)]

    for i, (top_left, bottom_right) in enumerate(coords):
        frame = frame_original[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]].copy()
        frame = np.array(frame, dtype=np.float32)

        if IMG_CHANNELS == 'Grayscale' or IMG_CHANNELS == 'Binarize':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
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

        if obj_class == 0:
            color = (0, 255, 0)
        elif obj_class == 1:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        cv2.rectangle(frame_original, top_left, bottom_right, color, 4)

        mirror = graph_background.copy()
        cv2.rectangle(mirror, (0, mirror.shape[0] - 1), (99, mirror.shape[0] - 1 - round(300 * confidence[0])), (0, 255, 0), -1)
        cv2.rectangle(mirror, (100, mirror.shape[0] - 1), (199, mirror.shape[0] - 1 - round(300 * confidence[1])), (0, 0, 255), -1)
        cv2.rectangle(mirror, (200, mirror.shape[0] - 1), (299, mirror.shape[0] - 1 - round(300 * confidence[2])), (255, 0, 0), -1)
        cv2.imshow(f'{"left" if i == 0 else "right"}_mirror', mirror)

    frame = background.copy()
    frame[0:IMG_HEIGHT, 0:IMG_WIDTH, :] = cv2.resize(frame_original[left_top_left[1]:left_bottom_right[1], left_top_left[0]:left_bottom_right[0]], (IMG_WIDTH, IMG_HEIGHT))
    frame[0:IMG_HEIGHT, IMG_WIDTH:IMG_WIDTH*2, :] = cv2.resize(frame_original[right_top_left[1]:right_bottom_right[1], right_top_left[0]:right_bottom_right[0]], (IMG_WIDTH, IMG_HEIGHT))
    cv2.imshow('frame', frame)

    cv2.waitKey(1)