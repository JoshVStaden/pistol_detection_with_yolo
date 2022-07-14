import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=1, C=1):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        """
        Defaults:
            C = 1
            B = 1
        Inputs:
        Predictions: Output of YOLO model. Shape (None, 2 * (C + B * 3))
        Target: Labelled targets. Shape: (None, C + 1 + (B * 4))

        Order of labels for Predictions:
            For each hand (2): [class_1,.., class_c, bbox1_w, bbox1_h, bbox1_conf,..., bboxb_conf]

        Order of labels for Target:
            [class_1,.., class_c, 1, bbox1_w, bbox1_h, bbox1_hand_x, bbox1_hand_y, ..., bboxb_hand_y]

        There is a 1 if there is an object there, 0 otherwise.
        """
        predictions = predictions.reshape(-1, 2, self.C + self.B * 3)

        ious = []
        for b in range(self.B):
            start_ind_pred = b * 3
            start_ind_target = b * 4
            iou_b = intersection_over_union(predictions[...,start_ind_pred + self.C:start_ind_pred + self.C + 2], target[..., start_ind_target + self.C + 1:start_ind_target + self.C + 3])
            ious.append(iou_b)

        # iou_b1 = intersection_over_union(predictions[...,self.C:self.C + 2], target[..., self.C + 1:self.C + 3])
        # iou_b2 = intersection_over_union(predictions[...,26:30], target[..., 21:25])
        # ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        ious = torch.cat([x.unsqueeze(0) for x in ious], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0)

        exists_box = target[..., self.C].unsqueeze(3) #Iobj_i

        # =================== #
        # BOX COORDS
        # =================== #
        b_pred = bestbox * predictions[..., self.C:self.C + 2]
        for b in range(1, self.B):
            ind = b * 3
            b_pred += (1 - bestbox) * predictions[..., ind + self.C: ind + self.C + 2]
    
        box_predictions = exists_box * b_pred
        box_targets = exists_box * target[...,self.C + 1:self.C + 3]

        # box_predictions[...,2:4] = (
        #     torch.sign(box_predictions[...,2:4])
        #     * torch.sqrt(torch.abs(box_predictions[...,2:4] + 1e-6))
        # )

        # box_targets[...,2:4] = torch.sqrt(box_targets[...,2:4])

        # (N, S, S, 4) -> (N * S * S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        
        # =================== #
        # OBJECT LOSS
        # =================== #
        pred_box = (
            bestbox * predictions[..., self.C:self.C + 2] 
        )
        for b in range(1, self.B):
            ind = b * 3
            pred_box += (1 - bestbox) * predictions[...,ind + self.C:ind + self.C + 2]
        # (N * S * S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[...,self.C:self.C + 1])
        )

        
        # =================== #
        # NO OBJECT LOSS
        # =================== #
        # (N, S, S, 1) -> (N, S * S)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C:self.C + 1], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C:self.C + 1], start_dim=1)
        )

        for b in range(1, self.B):
            ind = b * 3
            
            no_object_loss += self.mse(
                torch.flatten((1 - exists_box) * predictions[..., ind + self.C:ind + self.C + 1], start_dim=1),
                torch.flatten((1 - exists_box) * target[..., ind + self.C:ind + self.C + 1], start_dim=1)
            )

        
        # =================== #
        # CLASS LOSS
        # =================== #
        # (N, S,S, 20) -> (N * S * S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2),
            torch.flatten(exists_box * target[..., :self.C], end_dim=-2)
        )

        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss

