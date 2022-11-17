import torch
import torch.nn as nn


class IoUCoeff(nn.Module):
    def __init__(self, smooth=1e-6, thresh=0.45):
        super(IoUCoeff, self).__init__()
        self.smooth = smooth
        self.thresh = thresh

    def forward(self, inputs, targets):
        # [BATCH x 1 x H x W] --> [BATCH x H x W]
        inputs = inputs.squeeze(1)
        inputs = inputs > self.thresh
        targets = targets.squeeze(1)
        targets = torch.sigmoid(targets)
        targets = targets > self.thresh
        # Will be zero if Truth=0 or Prediction=0
        intersection = (targets & inputs).float().sum((1, 2))
        # Will be zero if both are 0
        union = (targets | inputs).float().sum((1, 2))
        # We smooth our division to avoid 0/0
        iou = (intersection + self.smooth) / (union + self.smooth)
        # This is equal to comparing with thresholds
        thresh = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10
        thresh = thresh.detach().cpu().numpy()
        return thresh


class DiceCoeff(nn.Module):
    def __init__(self, smooth=1):
        super(DiceCoeff, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # comment out if your model contains a sigmoid or equivalent
        # activation layer
        inputs = torch.sigmoid(inputs)
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_coeff = (2. * intersection + self.smooth) / \
                     (inputs.sum() + targets.sum() + self.smooth)
        return dice_coeff
