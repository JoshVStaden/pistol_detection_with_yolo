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






# for f in os.listdir(DATASET_DIR):
#     # print(f[-4:])
#     if f[-4:] != ".txt":
#         sample_name = f[:-4]
#         im_file = os.path.join(DATASET_DIR, f)
#         ann_file = os.path.join(DATASET_DIR, sample_name + ".txt")
#         img = cv2.imread(im_file)
#         with open(ann_file, 'r') as ann:
#             bbox = ann.readline().split(' ')
#             bbox = [float(x.strip()) for x in bbox]
#             bbox[1] = int(bbox[1] * img.shape[0])
#             bbox[2] = int(bbox[2] * img.shape[1])
#             bbox[3] = int(bbox[3] * img.shape[0])
#             bbox[3] = bbox[1] + bbox[3]
#             bbox[4] = int(bbox[4] * img.shape[1])
#             bbox[4] = bbox[2] + bbox[4]
            
#         print(img.shape)
#         print(bbox)
#         img = cv2.circle(img, (bbox[1], bbox[4]), 10, (255,0,0), -1)
#         img = cv2.rectangle(img, (bbox[1], bbox[2]), (bbox[3], bbox[4]), (0, 0, 255))
#         cv2.imshow(f, img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
