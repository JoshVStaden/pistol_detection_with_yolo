import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
from tqdm import tqdm
import matplotlib.patches as patches

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))



    return intersection / (box1_area + box2_area - intersection + 1e-6)


def width_height(box_pred, box_labels):
    w1 = torch.max(box_pred[...,0], box_labels[...,0])
    w2 = torch.min(box_pred[...,0], box_labels[...,0])
    h1 = torch.max(box_pred[...,1], box_labels[...,1])
    h2 = torch.min(box_pred[...,1], box_labels[...,1])

    intersection = (w1 - w2) * (h1 - h2)

    box1_area = abs(box_pred[...,0] * box_pred[...,1])
    box2_area = abs(box_labels[...,0] * box_labels[...,1])

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
    total_pred_boxes, total_true_boxes, iou_threshold=0.5, box_format="corners", num_classes=20
):
    """
    Calculates mean average precision 
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6


    for c in range(2):
        detections = []
        ground_truths = []

        tmp = []
        for b in total_pred_boxes:
            
            pred_boxes = [b[0]]
            pred_boxes.append(b[ 1 + c])
            pred_boxes.append(b[ 3 + c])
            for i in range(2):
                pred_boxes.append(b[ 5 + (c * 2) + i])
            for i in range(2):
                pred_boxes.append(b[ 9 + (c * 2) + i])
            tmp.append(pred_boxes)
        pred_boxes = tmp
        
        
        

        tmp = []
        for b in total_true_boxes:
            true_boxes = [b[0]]
            true_boxes.append(b[ 1 + c])
            true_boxes.append(b[ 3 + c])
            for i in range(2):
                true_boxes.append(b[ 5 + (c * 2) + i])
            for i in range(2):
                true_boxes.append(b[ 9 + (c * 2) + i])
            tmp.append(true_boxes)
        true_boxes = tmp
        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            
            if detection[1] == 1:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == 1:
                ground_truths.append(true_box)
    
        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            if val == 2:
                quit()
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        # detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                
                iou = intersection_over_union(
                    torch.tensor(detection[-4:]),
                    torch.tensor(gt[-4:]),
                    box_format=box_format,
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        
        
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        
        
        
        
        
        
        average_precisions.append(torch.trapz(precisions, recalls))
        
    
    # if sum(average_precisions) > 0:
    #     quit()
    return sum(average_precisions) / len(average_precisions)

def plot_image(image, boxes, filename="image.png", validation=False):
    """Plots predicted bounding boxes on the image"""
    base_dir = "val_examples/" if validation else "examples/"
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    boxes = boxes.numpy()
    for i in range(1):
        
        curr_box = boxes
        
        has_left_gun = curr_box[0] == 1
        has_right_gun = curr_box[1] == 1
        if has_left_gun:
                
            box = curr_box[[4, 5, 8, 9]]
            assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
            boxwidth = box[2] - box[0]
            boxheight = box[3] - box[1]

            upper_left_x = box[0] 
            upper_left_y = box[1]
            rect = patches.Rectangle(
                (upper_left_x * width, upper_left_y * height),
                boxwidth * width,
                boxheight * height,
                linewidth=1,
                edgecolor="g",
                facecolor="none",
            )
            # Add the patch to the Axes
            ax.add_patch(rect)
        if has_right_gun:
                
            box = curr_box[[6, 7, 10, 11]]
            assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
            boxwidth = box[2] - box[0]
            boxheight = box[3] - box[1]

            upper_left_x = box[0] 
            upper_left_y = box[1]
            rect = patches.Rectangle(
                (upper_left_x * width, upper_left_y * height),
                boxwidth * width,
                boxheight * height,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            # Add the patch to the Axes
            ax.add_patch(rect)

    plt.savefig(base_dir + filename)
    plt.close()

def example_images(x, labels, predictions):
    x, x_hands = x
    width, height = x.size()[-2], x.size()[-1]
    batch_size = x.size()[0]

    
    quit()
    for i in range(batch_size):
        x = x[i,...]
        x_hand = x_hands[i,...]
        pred = predictions[i, ...]
        pred_left = pred[:3]/2
        pred_right = pred[3:]/2
        x_pos_l = x_hands[0,0] - pred[1]
        y_pos_l = x_hands[0,1] - pred[2]
        x_pos_r = x_hands[1,0] + pred[4]
        y_pos_r = x_hands[1,1] + pred[5]
        plt.figure()
        plt.imshow(x)


    quit()

def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device=DEVICE,
    validation=False,
):
    # loop = tqdm(loader, leave=True)
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    im_no = 0
    for batch_idx, ((x, x_hands), labels) in enumerate(loader):
        x = x.to(device)
        x_hands = x_hands.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            predictions = model((x, x_hands))
        # if batch_idx == 0:
        #     example_images((x, x_hands), labels, predictions)
        batch_size = x.shape[0]
        
        
        true_bboxes = cellboxes_to_boxes(labels[...,:], x_hands=x_hands)
        bboxes = cellboxes_to_boxes(predictions, x_hands=x_hands)

        for idx in range(batch_size):
            # nms_boxes = non_max_suppression(
            #     bboxes[idx],
            #     iou_threshold=iou_threshold,
            #     threshold=threshold,
            #     box_format=box_format,
            # )
            nms_boxes = bboxes[idx]


            if True:# batch_idx == 0 and idx <= 4:
               plot_image(x[idx].permute(1,2,0).to("cpu"), true_bboxes[idx, ...], filename=f"label_{im_no}.png", validation=validation)
               plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes, filename=f"prediction_{im_no}.png", validation=validation)
               im_no += 1

            nms_boxes = nms_boxes.tolist()

            # for nms_box in nms_boxes:
            all_pred_boxes.append([train_idx] + nms_boxes)
            
            

            box = true_bboxes[idx]
            box = box.tolist()

            # for box in true_bboxes[idx]:
                # many will get converted to 0 pred
            # if box[1] > threshold:
            all_true_boxes.append([train_idx] + box)
            # loop.set_postfix(index=batch_idx, total=len(loop))

            train_idx += 1
    model.train()
    return all_pred_boxes, all_true_boxes



def _convert_target_cellbox(targets, S, C, B):
    """
    Convert Targets to cells

    Inputs:
        targets: shape (None, 2,  C  + (B * 3))
        Order of bboxes: (conf, w, h)
    """
    targets = targets.to("cpu")
    batch_size = targets.shape[0]
    targets = targets.reshape(batch_size, 2 *( (B * 3)))
    bboxes = []
    for b in range(2):
        ind = b * ( (B * 3))
        bboxes.append(targets[..., C + ind : C + ind + ( (B * 3)) ])

    best_boxes = bboxes[0]
    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)
    w = 1 / S * (best_boxes[..., 1:2] )
    h = 1 / S * (best_boxes[..., 3:] )
    converted_bboxes = torch.cat((w, h), dim=-1)
    predicted_class = targets[..., :C].argmax(-1).unsqueeze(-1)
    best_confidence = targets[..., C].unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds
    


def convert_cellboxes(predictions, S=1, C=1, B=1, x_hands=None):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.

    Each box contains 12 elements. In order:

    [left_class, right_class, left_conf, right_conf, l_xmin, l_ymin, r_xmin, r_ymin]
    # """
    # if x_hands is None:
    #     return _convert_target_cellbox(predictions, S, C, B)
    # predictions = predictions.to("cpu")
    # batch_size = predictions.shape[0]
    # predictions = predictions.reshape(batch_size, 2, (B * 3))
    # bboxes1 = predictions[...,  1:]
    # scores =predictions[..., 0].unsqueeze(0)
    # best_box = scores.argmax(0).unsqueeze(-1)
    # best_boxes = bboxes1 * (1 - best_box)
    # cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)
    # w_h = 1 / S * best_boxes[..., C:]

    # converted_bboxes = w_h
    # predicted_class = predictions[..., :C].argmax(-1).unsqueeze(-1)
    # best_confidence = predictions[..., -1].unsqueeze(-1)
    # converted_preds = torch.cat(
    #     (predicted_class, best_confidence, converted_bboxes), dim=-1
    # )
    
    

    # return converted_preds

    
    predictions = predictions.to("cpu")
    x_hands = x_hands.to("cpu")
    batch_size = predictions.shape[0]
    if predictions.size()[1] == 6:
        # print(predictions.size())
        predictions = torch.cat((predictions[...,:1], predictions[...,2:4], predictions[...,1:2], predictions[...,4:]), axis=-1)
        # print(predictions.size())
    predictions = predictions.reshape(batch_size, 2, (B * 3))
    
    

    bboxes_w = torch.cat((predictions[..., 0, 1:], predictions[..., 1, 1:]), dim=-1) / 2
    bboxes = x_hands[:,0,:]
    
    
    
    
    bboxes = torch.cat((bboxes - bboxes_w, bboxes + bboxes_w), dim=-1)
    # bboxes = bboxes - bboxes_w, bboxes + bboxes_w
    
    


    scores = torch.cat(
        (predictions[..., 0, 0].unsqueeze(0), predictions[...,1, 0].unsqueeze(0)), dim=0
    )
    # cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)
    # x = 1 / S * (bboxes[..., :1] + cell_indices)
    # y = 1 / S * (bboxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    # w_y = 1 / S * bboxes[..., 2:4]

    # converted_bboxes = torch.cat((x[...,0], y[...,0], w_y), dim=-1)
    converted_bboxes = bboxes
    predicted_class = torch.round(predictions[..., :, 0])
    best_confidence = predictions[..., :, 0]#torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
    #     -1
    # )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=1, x_hands=None):
    converted_pred = convert_cellboxes(out, x_hands=x_hands)#.reshape(out.shape[0], S * S, -1)
    return converted_pred
    
    
    # converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])