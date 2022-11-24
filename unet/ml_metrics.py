import torch
import torch.nn as nn


def get_metric_with_params(params):
    assert {"metric"}.issubset(params)
    metric_params = params.get("metric")
    assert {"type"}.issubset(metric_params)
    metric_type = metric_params.get("type")

    if metric_type.lower() == "ioucoeff":
        return IoUCoeff()

    if metric_type.lower() == "dicecoeff":
        return DiceCoeff()


class DiceCoeff(nn.Module):
    def __init__(self, smooth=1):
        super(DiceCoeff, self).__init__()
        self.smooth = smooth

    def forward(self, predicts, targets):
        # Apply activation. comment out if your model contains
        # a sigmoid or equivalent activation layer
        predicts = torch.sigmoid(predicts)
        # flatten label and prediction tensors
        predicts = predicts.view(-1)
        targets = targets.view(-1)
        intersection = (predicts * targets).sum()
        dice_coeff = (2. * intersection + self.smooth) / \
                     (predicts.sum() + targets.sum() + self.smooth)
        return dice_coeff


class IoUCoeff(nn.Module):
    def __init__(self, smooth=1e-6, thresh=0.45):
        super(IoUCoeff, self).__init__()
        self.smooth = smooth
        self.thresh = thresh

    def forward(self, predicts, targets):
        # Apply activation. comment out if your model contains
        # a sigmoid or equivalent activation layer
        predicts = torch.sigmoid(predicts)
        # Squeeze channel dim - (BATCH x 1 x H x W) --> (BATCH x H x W)
        predicts = predicts.squeeze(1)
        targets = targets.squeeze(1)
        # Make sure predictions and targets are binarized
        predicts = predicts > self.thresh
        targets = targets > self.thresh
        # Will be zero if Truth=0 or Prediction=0
        intersection = (predicts & targets).float().sum((1, 2))
        # Will be zero if both are 0
        union = (predicts | targets).float().sum((1, 2))
        # We smooth our division to avoid 0/0
        iou = (intersection + self.smooth) / (union + self.smooth)
        # This is equal to comparing with thresholds
        iou_score = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10
        return iou_score.detach().cpu().numpy()
