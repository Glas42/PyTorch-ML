from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import vgamepad as vg
import numpy as np
import bettercam
import time
import cv2
import os

camera = bettercam.create(output_color="BGR", output_idx=0)
gamepad = vg.VX360Gamepad()
lower_red = np.array([0, 0, 160])
upper_red = np.array([110, 110, 255])

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = ""
for file in os.listdir(SCRIPT_PATH):
    if file.endswith(".h5"):
        MODEL_PATH = os.path.join(SCRIPT_PATH, file)
        break
IMG_WIDTH = 420
IMG_HEIGHT = 220

def preprocess_image(frame):
    frame_pil = Image.fromarray(frame)
    frame_pil = frame_pil.resize((IMG_WIDTH, IMG_HEIGHT))
    frame_resized = np.array(frame_pil) / 255.0
    return frame_resized

model = load_model(MODEL_PATH)

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('frame', cv2.WND_PROP_TOPMOST, 1)
cv2.resizeWindow('frame', IMG_WIDTH, IMG_HEIGHT)

while True:
    start = time.time()
    frame = camera.grab()
    if type(frame) == type(None):
        continue
    frame = frame[759:979, 1479:1899]
    cv2.rectangle(frame, (0,0), (round(frame.shape[1]/6),round(frame.shape[0]/3)),(0,0,0),-1)
    cv2.rectangle(frame, (frame.shape[1],0), (round(frame.shape[1]-frame.shape[1]/6),round(frame.shape[0]/3)),(0,0,0),-1)
    frame = cv2.inRange(frame, lower_red, upper_red)
        
    frame = preprocess_image(frame)
    prediction = model.predict(np.expand_dims(frame, axis=0), verbose=0)[0][0]
    print("Predicted steering angle:", prediction)

    output = prediction * -0.05

    gamepad.left_joystick_float(x_value_float=output, y_value_float=0)
    gamepad.update()

    cv2.line(frame, (round(IMG_WIDTH * (output + 0.5)), 0), (round(IMG_WIDTH * (output + 0.5)), IMG_HEIGHT), (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {round(1 / (time.time() - start), 1)}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break