import torch

from unet.ml_models import UNet


def test_model():
    in_channels = 1
    out_classes = 1
    unet = UNet(n_channels=in_channels,
                n_classes=out_classes,
                bilinear=False)

    in_tensor_1 = torch.FloatTensor(2, 1, 80, 250)
    in_tensor_2 = torch.FloatTensor(4, 1, 100, 100)
    in_tensor_3 = torch.FloatTensor(6, 1, 264, 264)

    out_tensor_1 = unet(in_tensor_1)
    out_tensor_2 = unet(in_tensor_2)
    out_tensor_3 = unet(in_tensor_3)

    assert out_tensor_1.shape == (2, 1, 80, 250)
    assert out_tensor_2.shape == (4, 1, 100, 100)
    assert out_tensor_3.shape == (6, 1, 264, 264)
