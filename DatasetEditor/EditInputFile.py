import os

PATH = os.path.dirname(os.path.dirname(__file__)) + "\\ModelFiles\\"

count = len(os.listdir(f"{PATH}TrainingData")) / 2
curCount = 0
for file in os.listdir(f"{PATH}TrainingData"):
    if file.endswith(".txt"):
        line = open(os.path.join(f"{PATH}TrainingData", file), "r").readline()
        line = line.split(",")
        line = float(line[0])
        line = line * 300
        line = str(line)
        with open(os.path.join(f"{PATH}EditedTrainingData", file), "w") as f:
            f.truncate(0)
            f.write(line)
        curCount += 1
        if curCount % 10 == 0:
            print(f"{curCount}/{count} files edited")