import os

directory = r"C:\GitHub\NavigationDetectionAI\ModelFiles\TrainingData"
count = len(os.listdir(directory))
curCount = 0

for filename in os.listdir(directory):
    if os.path.isfile(os.path.join(directory, filename)):
        name, ext = os.path.splitext(filename)
        new_name = name.replace(".", "_") + ext
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
        curCount += 1
        if curCount % 10 == 0:
            print(f"{curCount}/{count} files edited")