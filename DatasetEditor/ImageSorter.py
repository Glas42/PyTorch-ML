import src.settings as settings
import keyboard
import time
import cv2
import os

images = []
last = time.time()
imgpersec = 50

dataset_path = settings.Get("DatasetPath", "unset")
if dataset_path == "unset" or not os.path.exists(dataset_path):
    print("Set DatasetPath in settings.json!")
    exit()

for file in os.listdir(dataset_path):
    if file.endswith(".png"):
        images.append((cv2.imread(os.path.join(f"{dataset_path}/{file}")), f"{dataset_path}/{file}"))
        
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