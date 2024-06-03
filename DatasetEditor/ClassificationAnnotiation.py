import keyboard
import cv2
import os

images = []
PATH = os.path.dirname(os.path.dirname(__file__)) + "\\ModelFiles\\"

go_forward = "space"
go_backward = "c"
classes = [("0", "a"), ("1", "s"), ("2", "d"), ("3", "f")] 
# 4 classes (0=red 1=yellow 2=green 3=nothing) and the corresponding keys to select the class while annotating

print("Creating image list... (May take a while, needs a lot of ram!)")
for file in os.listdir(f"{PATH}TrainingData"):
    if file.endswith(".png") and file not in os.listdir(f"{PATH}EditedTrainingData"):
        images.append((cv2.imread(os.path.join(f"{PATH}TrainingData/{file}")), f"{PATH}TrainingData/{file}"))

cv2.namedWindow('Classification - Annotiation', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Classification - Annotiation', 400, 800)

print("Please annotate the images!")
index = 0
while index < len(images):
    image, path = images[index]
    while True:

        cv2.imshow("Classification - Annotiation", image)
        cv2.waitKey(1)

        if keyboard.is_pressed(classes[0][1]):
            cv2.imwrite(os.path.join(f"{PATH}EditedTrainingData/{file}"), image)
            with open(os.path.join(f"{PATH}EditedTrainingData/{file.replace('.png', '.txt')}"), 'w') as f:
                f.write(classes[0][0])
                f.close()
            index += 1
            while keyboard.is_pressed(classes[0][1]):
                pass
            break

        if keyboard.is_pressed(classes[1][1]):
            cv2.imwrite(os.path.join(f"{PATH}EditedTrainingData/{file}"), image)
            with open(os.path.join(f"{PATH}EditedTrainingData/{file.replace('.png', '.txt')}"), 'w') as f:
                f.write(classes[1][0])
                f.close()
            index += 1
            while keyboard.is_pressed(classes[1][1]):
                pass
            break

        if keyboard.is_pressed(classes[2][1]):
            cv2.imwrite(os.path.join(f"{PATH}EditedTrainingData/{file}"), image)
            with open(os.path.join(f"{PATH}EditedTrainingData/{file.replace('.png', '.txt')}"), 'w') as f:
                f.write(classes[2][0])
                f.close()
            index += 1
            while keyboard.is_pressed(classes[2][1]):
                pass
            break

        if keyboard.is_pressed(classes[3][1]):
            cv2.imwrite(os.path.join(f"{PATH}EditedTrainingData/{file}"), image)
            with open(os.path.join(f"{PATH}EditedTrainingData/{file.replace('.png', '.txt')}"), 'w') as f:
                f.write(classes[3][0])
                f.close()
            index += 1
            while keyboard.is_pressed(classes[3][1]):
                pass
            break

        if keyboard.is_pressed(go_forward):
            index += 1
            while keyboard.is_pressed(go_forward):
                pass
            break
        if keyboard.is_pressed(go_backward):
            index -= 1
            while keyboard.is_pressed(go_backward):
                pass
            break