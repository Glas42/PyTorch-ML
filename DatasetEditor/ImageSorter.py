import keyboard
import shutil
import time
import cv2
import os

images = []
last = time.time()
imgpersec = 50

PATH = os.path.dirname(os.path.dirname(__file__)) + "\\ModelFiles\\"

print("Copying images...")
for file in os.listdir(f"{PATH}TrainingData"):
    if file.endswith(".png"):
        if not os.path.exists(os.path.join(f"{PATH}EditedTrainingData", file)):
            shutil.copy2(os.path.join(f"{PATH}TrainingData", file), os.path.join(f"{PATH}EditedTrainingData", file))

print("Creating image list... (May take a while, needs a lot of ram!)")
for file in os.listdir(f"{PATH}EditedTrainingData"):
    if file.endswith(".png"):
        images.append((cv2.imread(os.path.join(f"{PATH}EditedTrainingData/{file}")), f"{PATH}EditedTrainingData/{file}"))

print("Please sort the images!")
index = 0
while index < len(images):
    image, path = images[index]
    start = time.time()
    while True:
        cv2.imshow("image", image)
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
                os.remove(os.path.join(path.replace(".png", ".txt")))
            except Exception as ex:
                print(ex)
            index += 1
            break
        time.sleep(0.01)
    sleep_time = 1 / imgpersec - time.time() - start
    if sleep_time > 0:
        time.sleep(sleep_time)
    last = time.time()