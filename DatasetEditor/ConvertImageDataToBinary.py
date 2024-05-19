import src.settings as settings
import numpy as np
import cv2
import os

dataset_path = settings.Get("DatasetPath", "unset")
if dataset_path == "unset" or not os.path.exists(dataset_path):
    print("Set DatasetPath in settings.json!")
    exit()

lower = np.array([1, 1, 1])
upper = np.array([255, 255, 255])
count = 1

for file in os.listdir(f"{dataset_path}"):
    if file.endswith(".png"):
        img = cv2.imread(os.path.join(f"{dataset_path}", file))

        mask = cv2.inRange(img, lower, upper)

        cv2.imwrite(os.path.join(f"{dataset_path}", file), mask)
        
        count += 1
        if count % 10 == 0:
            print(f"{count} of {len(os.listdir(f"{dataset_path}"))//2}")