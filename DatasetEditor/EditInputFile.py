import cv2
import os

PATH = os.path.dirname(os.path.dirname(__file__)) + "\\ModelFiles\\"

count = len(os.listdir(f"{PATH}TrainingData")) / 2
curCount = 0

for file in os.listdir(f"{PATH}TrainingData"):
    if file.endswith(".png"):
        height, width = cv2.imread(os.path.join(f"{PATH}TrainingData", file)).shape[:2]
        break

for file in os.listdir(f"{PATH}TrainingData"):
    if file.endswith(".txt"):
        line = str(open(os.path.join(f"{PATH}TrainingData", file), "r").read()).replace("'", "").replace("(", "").replace(")", "").replace(" ", "").split(",")

        obj_x1 = float(int(line[0]) / width)
        obj_y1 = float(int(line[1]) / height)
        obj_x2 = float(int(line[2]) / width)
        obj_y2 = float(int(line[3]) / height)
        obj_class = line[4]

        line = f"{obj_x1},{obj_y1},{obj_x2},{obj_y2},{obj_class}"

        with open(os.path.join(f"{PATH}EditedTrainingData", file), "w") as f:
            f.truncate(0)
            f.write(line)
        curCount += 1
        if curCount % 10 == 0:
            print(f"{curCount}/{count} files edited")