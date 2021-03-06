import torch
import torch.nn as nn
from utils import intersection_over_union
from utils_pistol import width_height

from torch.nn import MSELoss


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class YoloLoss(nn.Module):
    def __init__(self, S=1, B=1, C=1):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    # def forward(self, predictions, target):
    #     """
    #     Defaults:
    #         C = 1
    #         B = 1
    #     Inputs:
    #     Predictions: Output of YOLO model. Shape (None, 2 * (C + B * 3))
    #     Target: Labelled targets. Shape: (None, C + 1 + (B * 4))

    #     Order of labels for Predictions:
    #         For each hand (2): [class_1,.., class_c, bbox1_w, bbox1_h, bbox1_conf,..., bboxb_conf]

    #     Order of labels for Target:
    #         [class_1,.., class_c, 1, bbox1_w, bbox1_h, bbox1_hand_x, bbox1_hand_y, ..., bboxb_hand_y]

    #     There is a 1 if there is an object there, 0 otherwise.
    #     """
        
    #     predictions = predictions.reshape(-1,self.S, self.S, 2, self.C + self.B * 3)

    #     # print(target.size())
    #     # quit()
    #     ltarget = target[..., :1, :]
    #     rtarget = target[...,1:, :]

    #     linds = (ltarget[...,self.C] == 1)[:, 0, 0, 0]
    #     if (linds).any():
    #         lious = width_height(predictions[linds, ...][...,:1, self.C + 1:self.C + 3], ltarget[linds, ...][..., self.C + 1:self.C + 3])
    #     else:
    #         lious = torch.ones((1,1,1,1), device=DEVICE)

    #     rinds = (rtarget[...,self.C] == 1)[:, 0, 0, 0]
    #     if (rinds).any():
    #         rious = width_height(predictions[rinds, ...][...,1:, self.C + 1:self.C + 3], rtarget[rinds, ...][..., self.C + 1:self.C + 3])
    #     else:
    #         rious = torch.ones((1,1,1,1), device=DEVICE)

    #     ious = torch.cat((lious, rious))

    #     # iou_b2 = width_height(predictions[...,26:30], target[..., 21:25])
    #     # ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
    #     iou_maxes, bestbox = torch.max(ious, dim=0)

    #     exists_box = target[..., self.C].unsqueeze(3) #Iobj_i

    #     # =================== #
    #     # BOX COORDS
    #     # =================== #
    #     box_predictions = exists_box * (
    #         # bestbox * predictions[..., 26:30]
    #          (1 - bestbox) * predictions[..., self.C + 1:self.C + 3]
    #     )
    #     box_targets = exists_box * target[...,self.C + 1:self.C + 3]

    #     box_predictions[...,2:4] = (
    #         torch.sign(box_predictions[...,2:4])
    #         * torch.sqrt(torch.abs(box_predictions[...,2:4] + 1e-6))
    #     )

    #     box_targets[...,2:4] = torch.sqrt(box_targets[...,2:4])

    #     # (N, S, S, 4) -> (N * S * S, 4)
    #     box_loss = self.mse(
    #         torch.flatten(box_predictions, end_dim=-2),
    #         torch.flatten(box_targets, end_dim=-2)
    #     )

        
    #     # =================== #
    #     # OBJECT LOSS
    #     # =================== #
    #     pred_box = (
    #         bestbox * predictions[..., self.C:self.C + 1] #+ (1 - bestbox) * predictions[...,20:21]
    #     )
    #     # (N * S * S)
    #     object_loss = self.mse(
    #         torch.flatten(exists_box * pred_box),
    #         torch.flatten(exists_box * target[...,self.C:self.C + 1])
    #     )

        
    #     # =================== #
    #     # NO OBJECT LOSS
    #     # =================== #
    #     # (N, S, S, 1) -> (N, S * S)
    #     no_object_loss = self.mse(
    #         torch.flatten((1 - exists_box) * predictions[..., self.C:self.C + 1], start_dim=1),
    #         torch.flatten((1 - exists_box) * target[..., self.C:self.C + 1], start_dim=1)
    #     )

    #     # no_object_loss += self.mse(
    #     #     torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
    #     #     torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
    #     # )

        
    #     # =================== #
    #     # CLASS LOSS
    #     # =================== #
    #     # (N, S,S, 20) -> (N * S * S, 20)
    #     class_loss = self.mse(
    #         torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2),
    #         torch.flatten(exists_box * target[..., :self.C], end_dim=-2)
    #     )

    #     # =================== #
    #     # HANDS LOSS
    #     # # =================== #
    #     # hand_box_predictions = exists_box * (
    #     #     bestbox * predictions[..., 26:28]
    #     #     + (1 - bestbox) * predictions[..., 21:23]
    #     # )
    #     # hand_box_targets = exists_box * target[...,30:32]

    #     # hand_box_predictions[...,2:4] = (
    #     #     torch.sign(hand_box_predictions[...,2:4])
    #     #     * torch.sqrt(torch.abs(hand_box_predictions[...,2:4] + 1e-6))
    #     # )

    #     # hand_box_targets[...,2:4] = torch.sqrt(hand_box_targets[...,2:4])



    #     # (N, S, S, 4) -> (N * S * S, 4)
    #     #hands_loss = 0#self.mse(
    #     #     torch.flatten(hand_box_predictions, end_dim=-1),
    #     #     torch.flatten(hand_box_targets, end_dim=-1)
    #     # )

    #     loss = (
    #         self.lambda_coord * box_loss
    #         # + object_loss
    #         # + self.lambda_noobj * no_object_loss
    #         + class_loss
    #         # + hands_loss
    #     )
        
    #     losses = (
    #         self.lambda_coord * box_loss,
    #         object_loss,
    #         self.lambda_noobj * no_object_loss,
    #         class_loss,
    #         # + hands_loss
    #     )

    #     return loss, losses

    def forward(self, predictions, target):
        """
        Defaults:
            C = 1
            B = 1
        Inputs:
        Predictions: Output of YOLO model. Shape (None, 2 * (B * 3))
        Target: Labelled targets. Shape: (None, C + 1 + (B * 4))

        Order of labels for Predictions:
            For each hand (2): [class_1,.., class_c, bbox1_w, bbox1_h, bbox1_conf,..., bboxb_conf]

        Order of labels for Target:
            [class_1,.., class_c, 1, bbox1_w, bbox1_h, bbox1_hand_x, bbox1_hand_y, ..., bboxb_hand_y]

        There is a 1 if there is an object there, 0 otherwise.
        """
        # print(target.size())
        # quit()
        left_target = target[:, 0,:]
        right_target = target[:, 1,:]

        left_pred = predictions[...,:3]
        right_pred = predictions[...,3:]
        # print(predictions.size())
        # quit()
        loss = MSELoss()
        # print(left_target.size(), left_pred.size())
        # quit()
        b = loss(left_target[...,1:], left_pred[...,1:]) + loss(right_target[...,1:], right_pred[...,1:])
        c = loss(left_target[...,0], left_pred[...,0]) + loss(right_target[...,0], right_pred[...,0])
        
        return b + c, (b,  c)