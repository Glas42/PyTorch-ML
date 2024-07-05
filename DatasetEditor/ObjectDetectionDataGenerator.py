import tensorflow as tf
import numpy as np
import cv2
import os

(images_train, labels_train), (images_test, labels_test) = tf.keras.datasets.mnist.load_data()

grid_size = 9
number_of_digits_per_image = 3
number_of_images_to_generate = 500
save_path = os.path.dirname(os.path.dirname(__file__)) + "\\ModelFiles\\EditedTrainingData"
save_path = save_path[:-1] if save_path[-1] in ["/", "\\"] else save_path

exit() if not os.path.exists(save_path) else None
exit() if input(f'Are you sure you want to generate {number_of_images_to_generate} images in the "{save_path}" folder? (y/n)\n-> ').lower() != "y" else None

background = np.zeros((grid_size * 28, grid_size * 28), dtype=np.uint8)

for _ in range(number_of_images_to_generate):
    frame = background.copy()
    annotation = []
    for i in range(number_of_digits_per_image):
        digit_placed = False
        while not digit_placed:
            x = np.random.randint(0, (grid_size - 1) * 28)
            y = np.random.randint(0, (grid_size - 1) * 28)
            index = np.random.randint(0, len(images_train))
            digit = images_train[index]
            label = labels_train[index]
            if np.sum(frame[y:(y+28), x:(x+28)]) == 0:
                min_x, min_y, max_x, max_y = float("inf"), float("inf"), 0, 0
                for row in range(28):
                    for col in range(28):
                        if digit[row][col] != 0:
                            if col < min_x:
                                min_x = col
                            if col > max_x:
                                max_x = col
                            if row < min_y:
                                min_y = row
                            if row > max_y:
                                max_y = row
                min_x += x
                min_y += y
                max_x += x
                max_y += y
                min_x /= frame.shape[1]
                min_y /= frame.shape[0]
                max_x /= frame.shape[1]
                max_y /= frame.shape[0]
                annotation.append(f"{min_x},{min_y},{max_x},{max_y}")
                frame[y:(y+28), x:(x+28)] = digit
                digit_placed = True

    name = len(os.listdir(save_path)) // 2

    with open(f"{save_path}/{name}.txt", "w") as f:
        f.write("\n".join(annotation))
        f.close()

    cv2.imwrite(f"{save_path}/{name}.png", frame)

    cv2.imshow("MNIST Grid", frame)
    cv2.waitKey(1)