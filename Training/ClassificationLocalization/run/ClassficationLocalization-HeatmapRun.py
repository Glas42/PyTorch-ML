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

metadata = {"data": []}
model = torch.jit.load(os.path.join(MODEL_PATH), _extra_files=metadata, map_location=device)
model.eval()

metadata = str(metadata["data"]).replace('b"(', '').replace(')"', '').replace("'", "").split(", ") # now in the format: ["key#value", "key#value", ...]
for var in metadata:
    if "outputs" in var:
        OUTPUTS = int(var.split("#")[1])
    if "image_width" in var:
        IMG_WIDTH = int(var.split("#")[1])
    if "image_height" in var:
        IMG_HEIGHT = int(var.split("#")[1])
    if "image_grayscale" in var:
        IMG_GRAYSCALE = True if var.split("#")[1] == "True" else False
    if "image_binarize" in var:
        IMG_BINARIZE = True if var.split("#")[1] == "True" else False
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

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('frame', cv2.WND_PROP_TOPMOST, 1)
cv2.resizeWindow('frame', 960, 540)

while True:
    start = time.time()
    frame = camera.grab()
    if frame is None:
        continue

    frame = np.array(frame)

    aspect_ratio = IMG_WIDTH / IMG_HEIGHT
    rows = 4
    image_height = frame.shape[0] // rows
    image_width = int(image_height * aspect_ratio)
    cols = frame.shape[1] // image_width

    with torch.no_grad():
        for i in range(rows):
            y1 = i * image_height
            y2 = (i + 1) * image_height
            for j in range(cols):
                x1 = j * image_width
                x2 = (j + 1) * image_width

                output = np.array(model(transform(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y1:y2, x1:x2] if IMG_GRAYSCALE else frame[y1:y2, x1:x2], (IMG_WIDTH, IMG_HEIGHT))).unsqueeze(0).to(device))[0].tolist())
                output = output * (1 / sum(output))
                confidence = [x / sum(output) for x in output]
                obj_class = np.argmax(output)
                obj_confidence = confidence[0] + confidence[1] + confidence[2]
                obj_confidence = obj_confidence / 3
                red = 1 if obj_class == 0 or obj_class == 1 else 0
                green = 1 if obj_class == 1 or obj_class == 2 else 0
                blue = 1 if obj_class == 3 else 0
                frame[y1:y2, x1:x2] = [round(obj_confidence * 127)*blue, round(obj_confidence * 127)*green, round(obj_confidence * 127)*red]

    cv2.putText(frame, f"FPS: {round(1 / (time.time() - start), 1)}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()