import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image

# Constants
PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_PATH = PATH + "\\ModelFiles\\EditedTrainingData"
MODEL_PATH = PATH + "\\ModelFiles\\Models"
IMG_HEIGHT = 220
IMG_WIDTH = 420
NUM_EPOCHS = 500
BATCH_SIZE = 128

# Function to load images and corresponding steering angles
def load_data(data_path):
    images = []
    user_inputs = []
    for file in os.listdir(data_path):
        if file.endswith(".png"):
            # Load image
            img = Image.open(os.path.join(data_path, file))
            img = img.resize((IMG_WIDTH, IMG_HEIGHT))
            img_array = np.array(img) / 255.0
            
            # Load steering angle if corresponding file exists
            user_inputs_file = os.path.join(data_path, file.replace(".png", ".txt"))
            if os.path.exists(user_inputs_file):
                with open(user_inputs_file, 'r') as f:
                    user_input = float(f.read().strip())
                images.append(img_array)
                user_inputs.append(user_input)
            else:
                print(f"Skipping file {file}: corresponding steering file not found.")
    
    return np.array(images), np.array(user_inputs)

# Load data
IMAGES, USERINPUTS = load_data(DATA_PATH)
print("Number of images loaded:", len(IMAGES))
print("Number of user inputs loaded:", len(USERINPUTS))

# Define model
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

# Train model
model.fit(IMAGES, USERINPUTS, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

# Save model
model.save(os.path.join(MODEL_PATH, f"EPOCHS-{NUM_EPOCHS}_BATCH-{BATCH_SIZE}_RES-{IMG_WIDTH}x{IMG_HEIGHT}_IMAGES-{len(X)}_DATE-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.h5"))
print("Model saved successfully.")