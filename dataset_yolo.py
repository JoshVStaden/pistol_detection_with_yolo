"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
from PIL import Image
import cv2
import xml.etree.ElementTree as ET
import mediapipe as mp
from tqdm import tqdm
import numpy as np


class PistolDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=1, B=1, C=1, transform=None, max_samples = 1500
    ):
        csv_file = csv_file + "_modified.txt"
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
        hands = []

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
        im_file = im_path#os.path.join(self.img_dir, im_path)
        image = cv2.cvtColor(cv2.imread(im_file), cv2.COLOR_BGR2RGB)
        
        # im_width, im_height, _ = image.shape

        image = Image.open(im_file)
        # Get Objects
        objects = label_root.iter("object")

        hand_arr = []
        hand_ind = 0

        lhand_arr = [0, 0]
        rhand_arr = [0, 0]


        for obj in objects:
            # [xmin, xmax, ymin, ymax]]
            curr_data = [0, 0, 0, 0, 0, 0]
            hand_arr = [0, 0, 0, 0]
            for child in obj:
                if child.tag == "rbbox":
                    curr_data[3] = 1
                    x_data = []
                    y_data = []
                    for bndbox_child in child:
                        if bndbox_child.tag == "xmin":
                            x_data.append(int(bndbox_child.text)/ im_width)
                        elif bndbox_child.tag == "xmax":
                            x_data.append(int(bndbox_child.text)/ im_width)
                        elif bndbox_child.tag == "ymax":
                            y_data.append(int(bndbox_child.text)/ im_height)
                        elif bndbox_child.tag == "ymin":
                            y_data.append(int(bndbox_child.text)/ im_height)

                        if len(x_data) == 2:
                            curr_data[4] = max(x_data) - min(x_data)
                        elif len(y_data) == 2:
                            curr_data[5] = max(y_data) - min(y_data)
                
                if child.tag == "lbbox":
                    curr_data[0] = 1
                    x_data = []
                    y_data = []
                    for bndbox_child in child:
                        if bndbox_child.tag == "xmin":
                            x_data.append(int(bndbox_child.text)/ im_width)
                        elif bndbox_child.tag == "xmax":
                            x_data.append(int(bndbox_child.text)/ im_width)
                        elif bndbox_child.tag == "ymax":
                            y_data.append(int(bndbox_child.text)/ im_height)
                        elif bndbox_child.tag == "ymin":
                            y_data.append(int(bndbox_child.text)/ im_height)

                        if len(x_data) == 2:
                            curr_data[2] = max(x_data) - min(x_data)
                        elif len(y_data) == 2:
                            curr_data[3] = max(y_data) - min(y_data)
                
                if child.tag == "lhand":
                    for l in child:
                        if l.tag == 'x':
                            hand_arr[0] = int(l.text)/ im_width
                        if l.tag == 'y':
                            hand_arr[1] = int(l.text)/ im_height

                if child.tag == "rhand":
                    for r in child:
                        if r.tag == 'x':
                            hand_arr[2] = int(r.text)/ im_width
                        if r.tag == 'y':
                            hand_arr[3] = int(r.text)/ im_height
            # has_lhand = lhand_arr == [0, 0]
            # has_rhand = rhand_arr == [0, 0]
            # hand_arr = [lhand_arr] + [rhand_arr]


            # if has_lhand and has_rhand:
            #     hand_arr = (lhand_arr, rhand_arr)
            # elif has_lhand:
            #     hand_arr = (lhand_arr, [0,0])
            # elif has_rhand:
            #     hand_arr = ([0,0], rhand_arr)
            hands.append(hand_arr)
            boxes.append(curr_data)
        
        boxes = torch.tensor(boxes)
        hand_arr = torch.Tensor(hands)
        print(boxes.size(), hand_arr.size())
        quit()

        image = (image, hand_arr)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        # Convert To Cells
        label_matrix = torch.zeros(( 2, self.B * 3))
        for box in boxes:
            # print(boxes)
            # quit()
            lclass_label, lwidth, lheight, rclass_label, rwidth, rheight = box.tolist()
            lclass_label = int(lclass_label)
            rclass_label = int(rclass_label)

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
            lwidth_cell, lheight_cell = (
                lwidth * self.S,
                lheight * self.S,
            )
            rwidth_cell, rheight_cell = (
                rwidth * self.S,
                rheight * self.S,
            )

            cell_sizes = [
                [lwidth_cell, lheight_cell],
                [rwidth_cell, rheight_cell]
            ]
            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!

            for i in range(2):
                hand_ind = i * 2 
                if box[hand_ind + 1] == 0 and box[hand_ind + 2] == 0:
                    continue
                if label_matrix[ i, 0] == 0:
                    # Set that there exists an object
                    label_matrix[ i, 0] = 1

                    

                    width_cell, height_cell = cell_sizes[i]

                    # Box coordinates
                    box_coordinates = torch.tensor(
                        [width_cell, height_cell]
                    )

                    label_matrix[ i, 1:3] = box_coordinates

                    # Set one hot encoding for class_label
                    label_matrix[i, class_label] = 1
                    print(label_matrix)
                    quit()
        return image, label_matrix

