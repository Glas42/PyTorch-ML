"""
Implementation of Yolo Loss Function from the original yolo paper

"""

import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    def __init__(self, SplitSize=7, BoundingBoxes=2, Classes=10):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        SplitSize is split size of image (in paper 7),
        BoundingBoxes is number of boxes (in paper 2),
        Classes is number of classes (in paper and VOC dataset is 20),
        """
        self.SplitSize = SplitSize
        self.BoundingBoxes = BoundingBoxes
        self.Classes = Classes

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, SplitSize*SplitSize(Classes+BoundingBoxes*5) when inputted
        predictions = predictions.reshape(-1, self.SplitSize, self.SplitSize, self.Classes + self.BoundingBoxes * 5)

        # Calculate IoU for the two predicted bounding boxes with target bbox
        iou_b1 = intersection_over_union(predictions[..., self.Classes+1:self.Classes+5], target[..., self.Classes+1:self.Classes+5])
        iou_b2 = intersection_over_union(predictions[..., self.Classes+6:self.Classes+10], target[..., self.Classes+1:self.Classes+5])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Take the box with highest IoU out of the two prediction
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., self.Classes].unsqueeze(3)  # in paper this is Iobj_i

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0. We only take out one of the two 
        # predictions, which is the one with highest Iou calculated previously.
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., self.Classes+6:self.Classes+10]
                + (1 - bestbox) * predictions[..., self.Classes+1:self.Classes+5]
            )
        )

        box_targets = exists_box * target[..., self.Classes+1:self.Classes+5]

        # Take sqrt of width, height of boxes to ensure that
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (
            bestbox * predictions[..., self.Classes+5:self.Classes+6] + (1 - bestbox) * predictions[..., self.Classes:self.Classes+1]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., self.Classes:self.Classes+1]),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        #max_no_obj = torch.max(predictions[..., 20:21], predictions[..., 25:26])
        #no_object_loss = self.mse(
        #    torch.flatten((1 - exists_box) * max_no_obj, start_dim=1),
        #    torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        #)

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.Classes:self.Classes+1], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.Classes:self.Classes+1], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.Classes+5:self.Classes+6], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.Classes:self.Classes+1], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.Classes], end_dim=-2,),
            torch.flatten(exists_box * target[..., :self.Classes], end_dim=-2,),
        )

        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        return loss
