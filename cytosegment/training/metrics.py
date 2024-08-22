import torch
import torch.nn as nn


def get_metric(config):
    """Retrieves a metric instance based on the provided configuration."""
    metric_name = config.train.metric.name.lower()

    if metric_name == "ioucoeff":
        return IoUCoeff()

    if metric_name == "dicecoeff":
        return DiceCoeff()


class DiceCoeff(nn.Module):
    """Initializes the DiceCoeff loss class."""

    def __init__(self, smooth=1e-7, thresh=0.5, sample_wise=False):
        super(DiceCoeff, self).__init__()
        self.smooth = smooth
        self.thresh = thresh
        self.sample_wise = sample_wise

    def forward(self, predicts, targets):
        # Apply thresholding
        predicts = torch.where(predicts >= self.thresh, 1.0, 0.0)
        targets = torch.where(targets >= self.thresh, 1.0, 0.0)

        if self.sample_wise:
            # Reshape predicts and targets [B, C, W, H] --> [B, C*W*H]
            predicts = predicts.view(predicts.shape[0], -1)
            targets = targets.view(targets.shape[0], -1)
            # Compute numerator and denominator
            numerator = 2 * (predicts * targets).sum(dim=1)
            denominator = predicts.sum(dim=1) + targets.sum(dim=1)
        else:
            # Reshape predicts and targets [B, C, W, H] --> [B*C*W*H]
            predicts = predicts.flatten()
            targets = targets.flatten()
            # Compute numerator and denominator
            numerator = 2 * (predicts * targets).sum()
            denominator = predicts.sum() + targets.sum()

        dice_coeff = numerator / (denominator + self.smooth)
        return dice_coeff


class IoUCoeff(nn.Module):
    """Initializes the IoUCoeff loss class."""

    def __init__(self, smooth=1e-6, thresh=0.5):
        super(IoUCoeff, self).__init__()
        self.smooth = smooth
        self.thresh = thresh

    def forward(self, predicts, targets):
        # Flatten and binarize the predictions and targets
        predicts = predicts.view(predicts.shape[0], -1) >= self.thresh
        targets = targets.view(targets.shape[0], -1) >= self.thresh

        # Compute intersection and union
        intersection = (predicts & targets).sum(dim=1)
        union = (predicts | targets).sum(dim=1)
        # Add smooth to avoid 0/0
        iou = (intersection + self.smooth) / (union + self.smooth)
        return iou
