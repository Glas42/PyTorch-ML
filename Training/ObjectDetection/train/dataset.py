from PIL import Image
import torch
import os

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir=None, split_size=None, boundingboxes=None, classes=None, transform=None):
        if data_dir is None or split_size is None or boundingboxes is None or classes is None or transform is None:
            raise "Function: __init__() of CustomDataset has missing parameters"
        self.data_dir = data_dir
        self.transform = transform
        self.split_size = split_size
        self.boundingboxes = boundingboxes
        self.classes = classes
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_name = self.files[index]
        img_path = os.path.join(self.data_dir, img_name)
        label_path = os.path.join(self.data_dir, img_name.replace('.jpg', '.txt'))
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [float(x) if float(x) != int(float(x)) else int(x) for x in label.replace("\n", "").split()]
                boxes.append([class_label, x, y, width, height])
        image = Image.open(img_path).convert("RGB")
        boxes = torch.tensor(boxes)
        image, boxes = self.transform(image, boxes)
        label_matrix = torch.zeros((self.split_size, self.split_size, self.classes + 5 * self.boundingboxes))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            i, j = int(self.split_size * y), int(self.split_size * x)
            x_cell, y_cell = self.split_size * x - j, self.split_size * y - i
            width_cell, height_cell = (width * self.split_size, height * self.split_size)
            if label_matrix[i, j, self.classes] == 0:
                label_matrix[i, j, self.classes] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, self.classes + 1:self.classes + 5] = box_coordinates
                label_matrix[i, j, class_label] = 1
        return image, label_matrix