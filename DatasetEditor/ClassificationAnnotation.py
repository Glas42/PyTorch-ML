import numpy as np
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


auto_annotation = input(f"Its possible to auto annotate the images using an existing model.\nTo do this place the model in the following path: {PATH}\n\nUse auto annotation? (y/n)\n-> ").lower() == "y"

if auto_annotation:
    auto_annotation_model = None
    for file in os.listdir(f"{PATH}"):
        if file.endswith(".pt"):
            print(f"\nFound a model: {file}\n")
            auto_annotation_model = file
    if auto_annotation_model == None:
        print("\nNo model found, auto annotation will not be available.\n")
    else:
        print("Trying to load model...")
        try:
            import torch
            from torchvision import transforms
            metadata = {"data": []}
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = torch.jit.load(os.path.join(f"{PATH}{auto_annotation_model}"), _extra_files=metadata, map_location=device)
            model.eval()
        except Exception as ex:
            print("\nError loading model, auto annotation will not be available.\n\nError message:\n" + str(ex) + "\n")
        CLASSES = None
        IMG_WIDTH = None
        IMG_HEIGHT = None
        IMG_CHANNELS = None
        metadata = str(metadata["data"]).replace('b"(', '').replace(')"', '').replace("'", "").split(", ")
        for var in metadata:
            if "classes" in var:
                CLASSES = int(var.split("#")[1])
            if "image_width" in var:
                IMG_WIDTH = int(var.split("#")[1])
            if "image_height" in var:
                IMG_HEIGHT = int(var.split("#")[1])
            if "image_channels" in var:
                IMG_CHANNELS = str(var.split("#")[1])
            if "transform" in var:
                transform = var.replace("\\n", "\n").replace('\\', '').split("#")[1]
                transform_list = []
                transform_parts = transform.strip().split("\n")
                for part in transform_parts[1:-1]:
                    part = part.strip()
                    if part:
                        try:
                            transform_args = []
                            transform_name = part.split("(")[0]
                            if "(" in part:
                                args = part.split("(")[1][:-1].split(",")
                                for arg in args:
                                    try:
                                        transform_args.append(int(arg.strip()))
                                    except ValueError:
                                        try:
                                            transform_args.append(float(arg.strip()))
                                        except ValueError:
                                            transform_args.append(arg.strip())
                            if transform_name == "ToTensor":
                                transform_list.append(transforms.ToTensor())
                            else:
                                transform_list.append(getattr(transforms, transform_name)(*transform_args))
                        except (AttributeError, IndexError, ValueError):
                            print(f"Skipping or failed to create transform: {part}")
                transform = transforms.Compose(transform_list)
        if CLASSES == None or IMG_WIDTH == None or IMG_HEIGHT == None or IMG_CHANNELS == None:
            print("Model metadata not found, auto annotation will not be available.\n")
        else:
            print("Model loaded successfully.\n")

print("\rCreating image list...", end="")
for i, file in enumerate(os.listdir(f"{PATH}EditedTrainingData")):
    if file.endswith(".png"):
        images.append((cv2.imread(os.path.join(f"{PATH}EditedTrainingData", file)), f"{file}"))
    if i % 100 == 0:
        print(f"\rCreating image list... ({round(i/len(os.listdir(f'{PATH}EditedTrainingData'))*100)}%)", end="")
print("\rCreated image list.           ", end="")

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
                os.remove(os.path.join(f"{PATH}EditedTrainingData/{file}"))
            except Exception as ex:
                print(ex)
            try:
                os.remove(os.path.join(f"{PATH}EditedTrainingData/{file.replace('.png', '.txt')}"))
            except Exception as ex:
                print(ex)
            index += 1
            continue


    predicted_class = None
    if auto_annotation:
        image = np.array(image, dtype=np.float32)
        if IMG_CHANNELS == 'Grayscale' or IMG_CHANNELS == 'Binarize':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if IMG_CHANNELS == 'RG':
            image = np.stack((image[:, :, 0], image[:, :, 1]), axis=2)
        elif IMG_CHANNELS == 'GB':
            image = np.stack((image[:, :, 1], image[:, :, 2]), axis=2)
        elif IMG_CHANNELS == 'RB':
            image = np.stack((image[:, :, 0], image[:, :, 2]), axis=2)
        elif IMG_CHANNELS == 'R':
            image = image[:, :, 0]
            image = np.expand_dims(image, axis=2)
        elif IMG_CHANNELS == 'G':
            image = image[:, :, 1]
            image = np.expand_dims(image, axis=2)
        elif IMG_CHANNELS == 'B':
            image = image[:, :, 2]
            image = np.expand_dims(image, axis=2)

        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image = image / 255.0

        if IMG_CHANNELS == 'Binarize':
            image = cv2.threshold(image, 0.5, 1.0, cv2.THRESH_BINARY)[1]


        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = np.array(model(image)[0].tolist())

        output = output * (1 / sum(output))
        confidence = [x / sum(output) for x in output]
        obj_class = np.argmax(output)
        obj_confidence = confidence[obj_class]
        if obj_confidence > 0.8:
            predicted_class = obj_class


    if predicted_class == None:
        class_0_color = (0, 0, 200)
        class_1_color = (0, 200, 200)
        class_2_color = (0, 200, 0)
        class_3_color = (200, 0, 0)
    else:
        class_0_color = (50, 50, 50)
        class_1_color = (50, 50, 50)
        class_2_color = (50, 50, 50)
        class_3_color = (50, 50, 50)
        if predicted_class == 0:
            class_0_color = (0, 0, 200)
        elif predicted_class == 1:
            class_1_color = (0, 200, 200)
        elif predicted_class == 2:
            class_2_color = (0, 200, 0)
        elif predicted_class == 3:
            class_3_color = (200, 0, 0)


    button_class_0_pressed, button_class_0_hovered = make_button(text="Red (Class 0)",
                                                            x1=0.52*frame_width,
                                                            y1=0.25*frame_height,
                                                            x2=0.98*frame_width,
                                                            y2=0.4*frame_height,
                                                            round_corners=30,
                                                            buttoncolor=class_0_color,
                                                            buttonhovercolor=(class_0_color[0]+20, class_0_color[1]+20, class_0_color[2]+20),
                                                            buttonselectedcolor=(class_0_color[0]+20, class_0_color[1]+20, class_0_color[2]+20),
                                                            textcolor=(255, 255, 255),
                                                            width_scale=0.95,
                                                            height_scale=0.5)

    button_class_1_pressed, button_class_1_hovered = make_button(text="Yellow (Class 1)",
                                                            x1=0.52*frame_width,
                                                            y1=0.425*frame_height,
                                                            x2=0.98*frame_width,
                                                            y2=0.575*frame_height,
                                                            round_corners=30,
                                                            buttoncolor=class_1_color,
                                                            buttonhovercolor=(class_1_color[0]+20, class_1_color[1]+20, class_1_color[2]+20),
                                                            buttonselectedcolor=(class_1_color[0]+20, class_1_color[1]+20, class_1_color[2]+20),
                                                            textcolor=(255, 255, 255),
                                                            width_scale=0.95,
                                                            height_scale=0.5)

    button_class_2_pressed, button_class_2_hovered = make_button(text="Green (Class 2)",
                                                            x1=0.52*frame_width,
                                                            y1=0.6*frame_height,
                                                            x2=0.98*frame_width,
                                                            y2=0.75*frame_height,
                                                            round_corners=30,
                                                            buttoncolor=class_2_color,
                                                            buttonhovercolor=(class_2_color[0]+20, class_2_color[1]+20, class_2_color[2]+20),
                                                            buttonselectedcolor=(class_2_color[0]+20, class_2_color[1]+20, class_2_color[2]+20),
                                                            textcolor=(255, 255, 255),
                                                            width_scale=0.95,
                                                            height_scale=0.5)

    button_class_3_pressed, button_class_3_hovered = make_button(text="Nothing (Class 3)",
                                                            x1=0.52*frame_width,
                                                            y1=0.83*frame_height,
                                                            x2=0.98*frame_width,
                                                            y2=0.98*frame_height,
                                                            round_corners=30,
                                                            buttoncolor=class_3_color,
                                                            buttonhovercolor=(class_3_color[0]+20, class_3_color[1]+20, class_3_color[2]+20),
                                                            buttonselectedcolor=(class_3_color[0]+20, class_3_color[1]+20, class_3_color[2]+20),
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


    if predicted_class == 0:
        button_class_0_pressed = True
    elif predicted_class == 1:
        button_class_1_pressed = True
    elif predicted_class == 2:
        button_class_2_pressed = True
    elif predicted_class == 3:
        button_class_3_pressed = True


    if button_class_0_pressed == True:
        with open(os.path.join(f"{PATH}EditedTrainingData/{file.replace('.png', '.txt')}"), 'w') as f:
            f.truncate(0)
            f.write("0")
            f.close()
        index += 1
    elif button_class_1_pressed == True:
        with open(os.path.join(f"{PATH}EditedTrainingData/{file.replace('.png', '.txt')}"), 'w') as f:
            f.truncate(0)
            f.write("1")
            f.close()
        index += 1
    elif button_class_2_pressed == True:
        with open(os.path.join(f"{PATH}EditedTrainingData/{file.replace('.png', '.txt')}"), 'w') as f:
            f.truncate(0)
            f.write("2")
            f.close()
        index += 1
    elif button_class_3_pressed == True:
        with open(os.path.join(f"{PATH}EditedTrainingData/{file.replace('.png', '.txt')}"), 'w') as f:
            f.truncate(0)
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