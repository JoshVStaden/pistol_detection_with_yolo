import cv2
import mediapipe as mp
import os

DATASET_DIR  = "../../Datasets/Guns_In_CCTV/VOC/"



train_files= []
val_files= []
test_files= []

dirs = ["train", "valid", "test"]
datas = [[], [] ,[]]

for i, d in enumerate(dirs):
    for f in os.listdir(DATASET_DIR + d + "/"):
        if f[-4:] == ".xml":
            datas[i].append(d + "/" + f[:-4])
    with open(f"CCTV/{d}.txt", "w") as outfile:
        outfile.write("\n".join(datas[i]))
