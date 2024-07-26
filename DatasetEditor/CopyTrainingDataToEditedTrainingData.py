import shutil
import cv2
import os

PATH = os.path.dirname(os.path.dirname(__file__)) + "\\ModelFiles\\"

overwrite = True

clear_folder = input("\nClear EditedTrainingData folder before copying? (y/n)\n-> ").lower() == "y"

if not clear_folder:
    overwrite = input("\nOverwrite existing images in the EditedTrainingData folder? (y/n)\n-> ").lower() == "y"

new_names = input("\nGive the files which are copied to the EditedTrainingData folder new names? (y/n)\n-> ").lower() == "y"

delete_corrupted_data = input("\nDelete corrupted images and their corresponding txt files? (y/n)\n-> ").lower() == "y"

if clear_folder:
    print("\nClearing folder...")
    for file in os.listdir(f"{PATH}EditedTrainingData"):
        os.remove(os.path.join(f"{PATH}EditedTrainingData", file))
    print("Folder cleared.")

print("\nCopying images...")
for file in os.listdir(f"{PATH}TrainingData"):
    if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
        if os.path.exists(os.path.join(f"{PATH}EditedTrainingData", file)) == False or overwrite:
            if new_names:
                name = str(len(os.listdir(f"{PATH}EditedTrainingData")) // (2 if os.path.exists(f"{PATH}TrainingData{file.replace(file.split(".")[-1], '.txt')}") else 1) + 1) + "." + file.split('.')[-1]
            shutil.copy2(os.path.join(f"{PATH}TrainingData", file), os.path.join(f"{PATH}EditedTrainingData", name if new_names else file.replace(".", "_")))
            try:
                shutil.copy2(os.path.join(f"{PATH}TrainingData", file.replace(file.split(".")[-1], ".txt")), os.path.join(f"{PATH}EditedTrainingData", name.replace(name.split(".")[-1], ".txt") if new_names else file.replace(file.split(".")[-1], ".txt").replace(".", "_")))
            except:
                pass
print("Images copied.")

corrupted_images = 0
if delete_corrupted_data:
    print("\nSearching for corrupted images...")
    for file in os.listdir(f"{PATH}EditedTrainingData"):
        if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
            try:
                img = None
                img = cv2.imread(os.path.join(f"{PATH}EditedTrainingData", file))
                if img is None:
                    try:
                        os.remove(os.path.join(f"{PATH}EditedTrainingData", file))
                    except:
                        pass
                    try:
                        os.remove(os.path.join(f"{PATH}EditedTrainingData", file.replace(file.split(".")[-1], ".txt")))
                    except:
                        pass
                    corrupted_images += 1
            except:
                try:
                    os.remove(os.path.join(f"{PATH}EditedTrainingData", file))
                except:
                    pass
                try:
                    os.remove(os.path.join(f"{PATH}EditedTrainingData", file.replace(file.split(".")[-1], ".txt")))
                except:
                    pass
                corrupted_images += 1
    print(f"{corrupted_images} Corrupted images deleted.")