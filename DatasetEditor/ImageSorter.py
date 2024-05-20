import keyboard
import time
import cv2
import os

images = []
last = time.time()
imgpersec = 50

PATH = os.path.dirname(os.path.dirname(__file__)) + "\\ModelFiles\\"

for file in os.listdir(f"{PATH}EditedTrainingData"):
    if file.endswith(".png"):
        images.append((cv2.imread(os.path.join(f"{PATH}EditedTrainingData/{file}")), f"{PATH}EditedTrainingData/{file}"))
        
for image, path in images:
    start = time.time()
    while True:
        cv2.imshow("image", image)
        cv2.waitKey(1)

        if keyboard.is_pressed('w'):
            print("approved")
            break
        if keyboard.is_pressed('s'):
            print("deleted")
            try:
                os.remove(os.path.join(path))
            except Exception as ex:
                print(ex)
            try:
                os.remove(os.path.join(path.replace(".png", ".txt")))
            except Exception as ex:
                print(ex)
            break
        time.sleep(0.01)

    sleep_time = 1 / imgpersec - time.time() - start
    if sleep_time > 0:
        time.sleep(sleep_time)
    
    last = time.time()