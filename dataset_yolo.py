"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

from __future__ import annotations
import torch
import os
import pandas as pd
from PIL import Image
import cv2
import xml.etree.ElementTree as ET
import mediapipe as mp
from tqdm import tqdm
import numpy as np

class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = (Image.open(img_path), torch.Tensor([0.5,0.5]))
        boxes = torch.tensor(boxes)

        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B + 2))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, 20] == 0:
                # Set that there exists an object
                label_matrix[i, j, 20] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates

                label_matrix[i, j, 25:27] = torch.Tensor([0.5, 0.5])
                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix

class MonashDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=1, B=1, C=1, transform=None, max_samples = 1500
    ):
        self.cached_file = csv_file.split(".")[0] +"_cached.pt"
        self.annotations = open(csv_file).read().split("\n")
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        self.max_samples = max_samples

        if os.path.exists(self.cached_file):
            self.cached_entries = torch.load(self.cached_file)
        else:
            self.mp_pose = mp.solutions.pose
            self.cached_entries = self.load_dataset()
        # print(len(self.cached_entries))
        # quit()
    def __len__(self):
        return len(self.cached_entries)

    def __getitem__(self, index):
        return self.cached_entries[index]

    def load_dataset(self):
        loop = tqdm(self.annotations, leave=True)
        ret = []
        for i, _ in enumerate(loop):
            if i == self.max_samples: 
                break
            ret.append(self.load_item(i))
            loop.set_postfix(loaded=i, total=len(self.annotations))
        # ret = ret
        torch.save(ret,self.cached_file)
        return ret


    def load_item(self, index):
        label_path = os.path.join(self.label_dir, self.annotations[index] + ".xml")
        boxes = []

        label_tree = ET.parse(label_path)
        label_root = label_tree.getroot()
        im_path, im_height, im_width = None, None, None
        for child in label_root:
            if child.tag == "filename":
                im_path = child.text
            if child.tag == "size":
                for size_child in child:
                    if size_child.tag == "width":
                        im_width = int(size_child.text)
                    elif size_child.tag == "height":
                        im_height = int(size_child.text)

        # Get Image File
        im_file = os.path.join(self.img_dir, im_path)
        # print(im_file)
        image = cv2.cvtColor(cv2.imread(im_file), cv2.COLOR_BGR2RGB)
        
        # im_width, im_height, _ = image.shape

        # Get Pose Information
        left_hand, right_hand = None, None
        with self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
            pose_information = pose.process(image)
            if pose_information.pose_landmarks:
                right_hand = (
                    int(pose_information.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].x ),
                    int(pose_information.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].y )
                )

                left_hand = (
                    int(pose_information.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].x ),
                    int(pose_information.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].y )
                )
            else:
                left_hand, right_hand = (0,0), (0,0)


        image = Image.open(im_file)
        # Get Objects
        objects = label_root.iter("object")

        hand_arr = None
        hand_ind = 0

        for obj in objects:
            # [xmin, xmax, ymin, ymax]
            curr_data = [-1, -1, -1, -1]
            for child in obj:
                if child.tag == "bndbox":
                    for bndbox_child in child:
                        if bndbox_child.tag == 'xmin':
                            curr_data[0] = int(bndbox_child.text)
                        elif bndbox_child.tag == 'xmax':
                            curr_data[1] = int(bndbox_child.text)
                        elif bndbox_child.tag == 'ymin':
                            curr_data[2] = int(bndbox_child.text)
                        elif bndbox_child.tag == 'ymax':
                            curr_data[3] = int(bndbox_child.text)
            xmin, xmax,ymin, ymax = curr_data

            x = ((xmin + xmax) / 2) / im_width
            width = (xmax - xmin) / im_width
            
            y = ((ymin + ymax) / 2) / im_height
            height = (ymax - ymin) / im_height

            left_x, left_y = left_hand
            right_x, right_y = right_hand

            xmin, xmax, ymin, ymax = xmin / im_width, xmax / im_width, ymin / im_height, ymax / im_height

            if xmin < left_x < xmax and ymin < left_y < ymax:
                hand_arr = [left_x, left_y]

            elif xmin < right_x < xmax and ymin < right_y < ymax:
                hand_ind = 1
                hand_arr = [right_x, right_y]

            else:
                hand_arr = [x, y]

            boxes.append([0, x, y, width, height])
        
        boxes = torch.tensor(boxes)
        hand_arr = torch.Tensor(hand_arr)
        # print(hand_arr)
        # quit()

        image = (image, hand_arr)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, 2, self.C + self.B * 3))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!

            if label_matrix[i, j, hand_ind, self.C] == 0:
                # Set that there exists an object
                label_matrix[i, j, hand_ind, self.C] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [width_cell, height_cell]
                )

                label_matrix[i, j, hand_ind, self.C + 1:self.C + 3] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j,hand_ind, class_label] = 1
        # print(label_matrix)
        # quit()
        return image, label_matrix

