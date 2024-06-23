import numpy as np
import cv2
import os

PATH = os.path.dirname(os.path.dirname(__file__)) + "\\ModelFiles\\"

lower = np.array([1, 1, 1])
upper = np.array([255, 255, 255])

for i, file in enumerate(os.listdir(f"{PATH}TrainingData")):
    if file.endswith(".png"):
        img = cv2.imread(os.path.join(f"{PATH}EditedTrainingData", file))

        mask = cv2.inRange(img, lower, upper)

        cv2.imwrite(os.path.join(f"{PATH}EditedTrainingData", file), mask)

        if i % 10 == 0:
            print(f"{len(os.listdir(f'{PATH}EditedTrainingData'))} of {len(os.listdir(f"{PATH}TrainingData"))//2}")