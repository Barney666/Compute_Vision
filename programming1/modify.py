import os
import shutil

val_classified = "val_classified/"


# os.mkdir(val_classified)
# for i in range(80):
#     os.mkdir(val_classified + str(i))

with open("val_anno.txt", "r") as f:
    for line in f.readlines():
        name = line.split(" ")[0]
        category = line.split(" ")[1].strip()
        path = val_classified + category
        shutil.copyfile("val/" + name, path + "/" + name)