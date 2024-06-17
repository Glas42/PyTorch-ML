import numpy as np
import shutil
import ctypes
import mouse
import cv2
import os

images = []
frame_width = 600
frame_height = 600
last_left_clicked = False
window_name = "Classification - Annotiation"
PATH = os.path.dirname(os.path.dirname(__file__)) + "\\ModelFiles\\"

classes = ["0", "1", "2", "3"]
# 4 classes (0=red 1=yellow 2=green 3=nothing)


def get_text_size(text="NONE", text_width=0.5*frame_width, max_text_height=0.5*frame_height):
    fontscale = 1
    textsize, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontscale, 1)
    width_current_text, height_current_text = textsize
    max_count_current_text = 3
    while width_current_text != text_width or height_current_text > max_text_height:
        fontscale *= min(text_width / textsize[0], max_text_height / textsize[1])
        textsize, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontscale, 1)
        max_count_current_text -= 1
        if max_count_current_text <= 0:
            break
    thickness = round(fontscale * 2)
    if thickness <= 0:
        thickness = 1
    return text, fontscale, thickness, textsize[0], textsize[1]


def make_button(text="NONE", x1=0, y1=0, x2=100, y2=100, round_corners=30, buttoncolor=(100, 100, 100), buttonhovercolor=(130, 130, 130), buttonselectedcolor=(160, 160, 160), buttonselected=False, textcolor=(255, 255, 255), width_scale=0.9, height_scale=0.8):
    if x1 <= mouse_x*frame_width <= x2 and y1 <= mouse_y*frame_height <= y2:
        buttonhovered = True
    else:
        buttonhovered = False
    if buttonselected == True:
        cv2.rectangle(frame, (round(x1+round_corners/2), round(y1+round_corners/2)), (round(x2-round_corners/2), round(y2-round_corners/2)), buttonselectedcolor, round_corners)
        cv2.rectangle(frame, (round(x1+round_corners/2), round(y1+round_corners/2)), (round(x2-round_corners/2), round(y2-round_corners/2)), buttonselectedcolor, -1)
    elif buttonhovered == True:
        cv2.rectangle(frame, (round(x1+round_corners/2), round(y1+round_corners/2)), (round(x2-round_corners/2), round(y2-round_corners/2)), buttonhovercolor, round_corners)
        cv2.rectangle(frame, (round(x1+round_corners/2), round(y1+round_corners/2)), (round(x2-round_corners/2), round(y2-round_corners/2)), buttonhovercolor, -1)
    else:
        cv2.rectangle(frame, (round(x1+round_corners/2), round(y1+round_corners/2)), (round(x2-round_corners/2), round(y2-round_corners/2)), buttoncolor, round_corners)
        cv2.rectangle(frame, (round(x1+round_corners/2), round(y1+round_corners/2)), (round(x2-round_corners/2), round(y2-round_corners/2)), buttoncolor, -1)
    text, fontscale, thickness, width, height = get_text_size(text, round((x2-x1)*width_scale), round((y2-y1)*height_scale))
    cv2.putText(frame, text, (round(x1 + (x2-x1) / 2 - width / 2), round(y1 + (y2-y1) / 2 + height / 2)), cv2.FONT_HERSHEY_SIMPLEX, fontscale, textcolor, thickness, cv2.LINE_AA)
    if x1 <= mouse_x*frame_width <= x2 and y1 <= mouse_y*frame_height <= y2 and left_clicked == False and last_left_clicked == True:
        return True, buttonhovered
    else:
        return False, buttonhovered


print("Creating image list... (May take a while, needs a lot of ram!)")
for file in os.listdir(f"{PATH}TrainingData"):
    if file.endswith(".png") and file not in os.listdir(f"{PATH}EditedTrainingData"):
        images.append((cv2.imread(os.path.join(f"{PATH}TrainingData/{file}")), f"{file}"))


background = np.zeros((frame_height, frame_width, 3), np.uint8)
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, frame_width, frame_height)
cv2.imshow(window_name, background)
cv2.waitKey(1)
if os.name == "nt":
    import win32gui
    from ctypes import windll, byref, sizeof, c_int
    hwnd = win32gui.FindWindow(None, window_name)
    windll.dwmapi.DwmSetWindowAttribute(hwnd, 35, byref(c_int(0x000000)), sizeof(c_int))

print("Please annotate the images!")
index = 0
while True:

    image, file = images[index]

    frame = background.copy()
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    if ctypes.windll.user32.GetKeyState(0x01) & 0x8000 != 0 and ctypes.windll.user32.GetForegroundWindow() == ctypes.windll.user32.FindWindowW(None, window_name):
        left_clicked = True
    else:
        left_clicked = False

    try:
        window_x, window_y, window_width, window_height = cv2.getWindowImageRect(window_name)
        mouse_x, mouse_y = mouse.get_position()
        mouse_relative_window = mouse_x-window_x, mouse_y-window_y
        last_window_size = (window_x, window_y, window_width, window_height)
        last_mouse_position = (mouse_x, mouse_y)
    except:
        try:
            window_x, window_y, window_width, window_height = last_window_size
        except:
            window_x, window_y, window_width, window_height = (0, 0, 100, 100)
        try:
            mouse_x, mouse_y = last_mouse_position
        except:
            mouse_x, mouse_y = (0, 0)
        mouse_relative_window = mouse_x-window_x, mouse_y-window_y
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, frame_width, frame_height)
        if os.name == "nt":
            hwnd = win32gui.FindWindow(None, window_name)
            windll.dwmapi.DwmSetWindowAttribute(hwnd, 35, byref(c_int(0x000000)), sizeof(c_int))

    if window_width != 0 and window_height != 0:
        mouse_x = mouse_relative_window[0]/window_width
        mouse_y = mouse_relative_window[1]/window_height
    else:
        mouse_x = 0
        mouse_y = 0

    try:
        image_resized = cv2.resize(image, (frame_width//2, frame_height))
        frame[0:image_resized.shape[0], 0:image_resized.shape[1]] = image_resized
    except:
        print("Error resizing image:", file)
        if input("Delete this image? (y/n)").lower() == "y":
            try:
                os.remove(os.path.join(f"{PATH}TrainingData/{file}"))
            except Exception as ex:
                print(ex)
            try:
                os.remove(os.path.join(f"{PATH}TrainingData/{file.replace('.png', '.txt')}"))
            except Exception as ex:
                print(ex)
            index += 1
            continue

    button_class_0_pressed, button_class_0_hovered = make_button(text="Red (Class 0)",
                                                            x1=0.52*frame_width,
                                                            y1=0.25*frame_height,
                                                            x2=0.98*frame_width,
                                                            y2=0.4*frame_height,
                                                            round_corners=30,
                                                            buttoncolor=(0, 0, 200),
                                                            buttonhovercolor=(20, 20, 220),
                                                            buttonselectedcolor=(20, 20, 220),
                                                            textcolor=(255, 255, 255),
                                                            width_scale=0.95,
                                                            height_scale=0.5)

    button_class_1_pressed, button_class_1_hovered = make_button(text="Yellow (Class 1)",
                                                            x1=0.52*frame_width,
                                                            y1=0.425*frame_height,
                                                            x2=0.98*frame_width,
                                                            y2=0.575*frame_height,
                                                            round_corners=30,
                                                            buttoncolor=(0, 200, 200),
                                                            buttonhovercolor=(20, 220, 220),
                                                            buttonselectedcolor=(20, 220, 220),
                                                            textcolor=(255, 255, 255),
                                                            width_scale=0.95,
                                                            height_scale=0.5)

    button_class_2_pressed, button_class_2_hovered = make_button(text="Green (Class 2)",
                                                            x1=0.52*frame_width,
                                                            y1=0.6*frame_height,
                                                            x2=0.98*frame_width,
                                                            y2=0.75*frame_height,
                                                            round_corners=30,
                                                            buttoncolor=(0, 200, 0),
                                                            buttonhovercolor=(20, 220, 20),
                                                            buttonselectedcolor=(20, 220, 20),
                                                            textcolor=(255, 255, 255),
                                                            width_scale=0.95,
                                                            height_scale=0.5)

    button_class_3_pressed, button_class_3_hovered = make_button(text="Nothing (Class 3)",
                                                            x1=0.52*frame_width,
                                                            y1=0.83*frame_height,
                                                            x2=0.98*frame_width,
                                                            y2=0.98*frame_height,
                                                            round_corners=30,
                                                            buttoncolor=(200, 0, 0),
                                                            buttonhovercolor=(220, 20, 20),
                                                            buttonselectedcolor=(220, 20, 20),
                                                            textcolor=(255, 255, 255),
                                                            width_scale=0.95,
                                                            height_scale=0.5)

    button_back_pressed, button_back_hovered = make_button(text="<--",
                                                            x1=0.52*frame_width,
                                                            y1=0.02*frame_height,
                                                            x2=0.74*frame_width,
                                                            y2=0.17*frame_height,
                                                            round_corners=30,
                                                            buttoncolor=(200, 0, 0),
                                                            buttonhovercolor=(220, 20, 20),
                                                            buttonselectedcolor=(220, 20, 20),
                                                            textcolor=(255, 255, 255),
                                                            width_scale=0.95,
                                                            height_scale=0.5)

    button_forward_pressed, button_forward_hovered = make_button(text="-->",
                                                            x1=0.76*frame_width,
                                                            y1=0.02*frame_height,
                                                            x2=0.98*frame_width,
                                                            y2=0.17*frame_height,
                                                            round_corners=30,
                                                            buttoncolor=(200, 0, 0),
                                                            buttonhovercolor=(220, 20, 20),
                                                            buttonselectedcolor=(220, 20, 20),
                                                            textcolor=(255, 255, 255),
                                                            width_scale=0.95,
                                                            height_scale=0.5)


    if button_class_0_pressed == True:
        cv2.imwrite(os.path.join(f"{PATH}EditedTrainingData/{file}"), image)
        with open(os.path.join(f"{PATH}EditedTrainingData/{file.replace('.png', '.txt')}"), 'w') as f:
            f.write("0")
            f.close()
        index += 1
    elif button_class_1_pressed == True:
        cv2.imwrite(os.path.join(f"{PATH}EditedTrainingData/{file}"), image)
        with open(os.path.join(f"{PATH}EditedTrainingData/{file.replace('.png', '.txt')}"), 'w') as f:
            f.write("1")
            f.close()
        index += 1
    elif button_class_2_pressed == True:
        cv2.imwrite(os.path.join(f"{PATH}EditedTrainingData/{file}"), image)
        with open(os.path.join(f"{PATH}EditedTrainingData/{file.replace('.png', '.txt')}"), 'w') as f:
            f.write("2")
            f.close()
        index += 1
    elif button_class_3_pressed == True:
        cv2.imwrite(os.path.join(f"{PATH}EditedTrainingData/{file}"), image)
        with open(os.path.join(f"{PATH}EditedTrainingData/{file.replace('.png', '.txt')}"), 'w') as f:
            f.write("3")
            f.close()
        index += 1
    elif button_back_pressed == True:
        if index > 0:
            index -= 1
        else:
            index = 0
    elif button_forward_pressed == True:
        if index < len(images) - 1:
            index += 1
        else:
            print("Done!")
            break

    last_left_clicked = left_clicked

    cv2.imshow(window_name, frame)
    cv2.waitKey(1)