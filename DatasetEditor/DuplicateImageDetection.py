import time
import cv2
import os

PATH = os.path.dirname(os.path.dirname(__file__)) + "\\ModelFiles\\"

GRAYSCALE = True
IMG_WIDTH = 90
IMG_HEIGHT = 150
MAX_PIXEL_DIFF = 10
MIN_SIMILARITY = 0.9
MIN_SHAPE_SIMILARITY = 0.95

total_files = len(os.listdir(f'{PATH}EditedTrainingData'))

images = []
start = time.time()
print("\rLoading Images...", end="")
for i, file in enumerate(os.listdir(f"{PATH}EditedTrainingData")):
    if file.endswith(".png"):
        image = cv2.imread(os.path.join(f"{PATH}EditedTrainingData", file))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if GRAYSCALE else image
        shape = image.shape
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        images.append((image, shape, f"{PATH}EditedTrainingData\\{file}"))
    if i % 100 == 0:
        eta = time.strftime('%H:%M:%S', time.gmtime(round((time.time() - start) / (i + 1) * total_files - (time.time() - start) + (time.time() - time.time()), 2)))
        print(f"\rLoading Images: {round(100 * i / total_files)}% ({eta})", end="")
print("\rLoading Images: 100%           \n\n", end="")

start = time.time()
duplicate_images = []
print("\rSearching for duplicates...", end="")
for i, (image1, shape1, file1) in enumerate(images):
    for j, (image2, shape2, file2) in enumerate(images):
        if file1 != file2:
            if abs(shape1[0] - shape2[0]) <= (shape1[0] + shape2[0]) / 2 * (1 - MIN_SHAPE_SIMILARITY) and abs(shape1[1] - shape2[1]) <= (shape1[1] + shape2[1]) / 2 * (1 - MIN_SHAPE_SIMILARITY):
                diff = cv2.absdiff(image1, image2)
                if cv2.countNonZero(diff[diff > MAX_PIXEL_DIFF]) / (image1.shape[0] * image1.shape[1]) <= 1 - MIN_SIMILARITY:
                    duplicate_images.append((file2))
    if i % 100 == 0:
        eta = time.strftime('%H:%M:%S', time.gmtime(round((time.time() - start) / (i + 1) * total_files - (time.time() - start) + (time.time() - time.time()), 2)))
        print(f"\rSearching for duplicates: {round(100 * i / total_files)}% ({eta})", end="")
print("\rSearching for duplicates: 100%           \n\n", end="")

start = time.time()
duplicate_images = list(set(duplicate_images))
print("\rDeleting images and their corresponding txt files...", end="")
for i, file in enumerate(duplicate_images):
    try:
        os.remove(os.path.join(file))
    except:
        pass
    try:
        os.remove(os.path.join(file.replace(".png", ".txt")))
    except:
        pass
    if i % 100 == 0:
        eta = time.strftime('%H:%M:%S', time.gmtime(round((time.time() - start) / (i + 1) * total_files - (time.time() - start) + (time.time() - time.time()), 2)))
        print(f"\rDeleting images and their corresponding txt files: {round(100 * i / len(duplicate_images))}% ({eta})", end="")
print("\rDeleting images and their corresponding txt files: 100%           ", end="")