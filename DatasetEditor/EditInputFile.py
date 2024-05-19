import src.settings as settings
import os

dataset_path = settings.Get("DatasetPath", "unset")
if dataset_path == "unset" or not os.path.exists(dataset_path):
    print("Set DatasetPath in settings.json!")
    exit()

count = len(os.listdir(f"{dataset_path}")) / 2
curCount = 0
for file in os.listdir(f"{dataset_path}"):
    if file.endswith(".txt"):
        line = open(os.path.join(f"{dataset_path}", file), "r").readline()
        line = line.split(",")
        line = float(line[0])
        line = line * 300
        line = str(line)
        with open(os.path.join(f"{dataset_path}", file), "w") as f:
            f.truncate(0)
            f.write(line)
        curCount += 1
        if curCount % 10 == 0:
            print(f"{curCount}/{count} files edited")