import keyboard
import shutil
import time
import cv2
import os

images = []
imgpersec = -1
dataset_for_object_detection = True

PATH = os.path.dirname(os.path.dirname(__file__)) + "\\ModelFiles\\"

print("Copying images...")
for file in os.listdir(f"{PATH}TrainingData"):
    if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
        if not os.path.exists(os.path.join(f"{PATH}EditedTrainingData", file)):
            shutil.copy2(os.path.join(f"{PATH}TrainingData", file), os.path.join(f"{PATH}EditedTrainingData", file))

print("Creating image list... (May take a while, needs a lot of ram!)")
if dataset_for_object_detection == False:
    for file in os.listdir(f"{PATH}EditedTrainingData"):
        if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
            images.append((cv2.imread(os.path.join(f"{PATH}EditedTrainingData/{file}")), f"{PATH}EditedTrainingData/{file}"))
else:
    for file in os.listdir(f"{PATH}EditedTrainingData"):
        if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
            with open(os.path.join(f"{PATH}EditedTrainingData/{file.replace(file.split(".")[-1], '.txt')}"), 'r') as f:
                content = f.read().split(',')
                x1, y1, x2, y2 = float(content[0]), float(content[1]), float(content[2]), float(content[3])
            images.append((cv2.imread(os.path.join(f"{PATH}EditedTrainingData/{file}")), (x1, y1, x2, y2, str(content[4])), f"{PATH}EditedTrainingData/{file}"))

print("Please sort the images!")
index = 0
while index < len(images):
    if dataset_for_object_detection == False:
        image, path = images[index]
    else:
        image, box, path = images[index]
        cv2.rectangle(image, (int(image.shape[1] * box[0]), int(image.shape[0] * box[1])), (int(image.shape[1] * box[2]), int(image.shape[0] * box[3])), (0, 0, 255) if box[4] == "Red" else (0, 255, 255) if box[4] == "Yellow" else (0, 255, 0), 1)
    start = time.time()
    while True:
        cv2.imshow("press 'w' to approve, 's' to back, 'space' to delete", image)
        cv2.waitKey(1)
        if keyboard.is_pressed('w'):
            print("approved")
            index += 1
            break
        if keyboard.is_pressed('s'):
            print("back")
            if index > 0:
                index -= 1
                break
        if keyboard.is_pressed('space'):
            print("deleted")
            try:
                os.remove(os.path.join(path))
            except Exception as ex:
                print(ex)
            try:
                os.remove(os.path.join(path.replace(path.split(".")[-1], ".txt")))
            except Exception as ex:
                print(ex)
            index += 1
            break
        time.sleep(0.01)
    if imgpersec > 0:
        sleep_time = 1 / imgpersec - time.time() - start
        if sleep_time > 0:
            time.sleep(sleep_time)
    else:
        while keyboard.is_pressed('w'): pass
        while keyboard.is_pressed('s'): pass
        while keyboard.is_pressed('space'): pass