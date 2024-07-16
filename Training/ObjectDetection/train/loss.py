import torch
import torch.nn as nn
from utils import intersection_over_union

class Loss(nn.Module):
    def __init__(self, split_size=None, boundingboxes=None, classes=None):
        if split_size is None or boundingboxes is None or classes is None:
            raise "Function: __init__() of Loss has missing parameters"
        super(Loss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.split_size = split_size
        self.boundingboxes = boundingboxes
        self.classes = classes

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions=None, target=None):
        if predictions is None or target is None:
            raise "Function: forward() of Loss has missing parameters"
        predictions = predictions.reshape(-1, self.split_size, self.split_size, self.classes + self.boundingboxes * 5)
        iou_b1 = intersection_over_union(predictions[..., self.classes + 1:self.classes + 5], target[..., self.classes + 1:self.classes + 5])
        iou_b2 = intersection_over_union(predictions[..., self.classes + 6:self.classes + 10], target[..., self.classes + 1:self.classes + 5])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., self.classes].unsqueeze(3)
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., self.classes + 6:self.classes + 10]
                + (1 - bestbox) * predictions[..., self.classes + 1:self.classes + 5]
            )
        )
        box_targets = exists_box * target[..., self.classes + 1:self.classes + 5]
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )
        pred_box = (
            bestbox * predictions[..., self.classes + 5:self.classes + 6] + (1 - bestbox) * predictions[..., self.classes:self.classes + 1]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., self.classes:self.classes + 1]),
        )
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.classes:self.classes + 1], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.classes:self.classes + 1], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.classes + 5:self.classes + 6], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.classes:self.classes + 1], start_dim=1)
        )
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.classes], end_dim=-2,),
            torch.flatten(exists_box * target[..., :self.classes], end_dim=-2,),
        )
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
        print(f"\rObject Loss: {object_loss}, Class Loss: {class_loss}, Box Loss: {box_loss}, No Object Loss: {no_object_loss}", end="", flush=True)
        return loss