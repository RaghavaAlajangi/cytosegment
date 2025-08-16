import torch
import torch.nn as nn
import torch.nn.functional as fun


class DoubleConv(nn.Module):
    """Double Convolution Block with optional intermediate channels."""

    def __init__(self, **kwargs):
        super().__init__()
        in_channels = kwargs.get("in_channels")
        out_channels = kwargs.get("out_channels")
        mid_channels = kwargs.get("mid_channels", out_channels)

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, in_tensor):
        return self.double_conv(in_tensor)


class Down(nn.Module):
    """Downscaling block"""

    def __init__(self, **kwargs):
        super().__init__()
        in_channels = kwargs.get("in_channels")
        out_channels = kwargs.get("out_channels")

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels=in_channels, out_channels=out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block"""

    def __init__(self, **kwargs):
        super().__init__()
        in_channels = kwargs.get("in_channels")
        out_channels = kwargs.get("out_channels")
        bilinear = kwargs.get("bilinear", True)

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.conv = DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                mid_channels=in_channels // 2,
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size=(2, 2),
                stride=(2, 2),
            )
            self.conv = DoubleConv(
                in_channels=in_channels, out_channels=out_channels
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Calculate padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Pad x1
        x1 = fun.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        )

        # Concatenate and apply convolution
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output Convolution Block."""

    def __init__(self, **kwargs):
        super(OutConv, self).__init__()
        in_channels = kwargs.get("in_channels")
        out_channels = kwargs.get("out_channels")

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        return self.conv(x)


class BenchmarkUNet(nn.Module):
    """U-Net architecture for image segmentation.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_classes : int
        Number of output classes.
    bilinear : bool
        If True, uses bilinear upsampling. Defaults to False.

    Note:
        Find the code here: https://github.com/xiaopeng-liao/Pytorch-UNet
    """

    def __init__(self, **kwargs):
        super(BenchmarkUNet, self).__init__()
        in_channels = kwargs.get("in_channels", 1)
        out_classes = kwargs.get("out_classes", 1)
        bilinear = kwargs.get("bilinear", False)

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_channels=in_channels, out_channels=64)
        self.down1 = Down(in_channels=64, out_channels=128)
        self.down2 = Down(in_channels=128, out_channels=256)
        self.down3 = Down(in_channels=256, out_channels=512)
        self.down4 = Down(in_channels=512, out_channels=1024 // factor)
        self.up1 = Up(
            in_channels=1024, out_channels=512 // factor, bilinear=bilinear
        )
        self.up2 = Up(
            in_channels=512, out_channels=256 // factor, bilinear=bilinear
        )
        self.up3 = Up(
            in_channels=256, out_channels=128 // factor, bilinear=bilinear
        )
        self.up4 = Up(in_channels=128, out_channels=64, bilinear=bilinear)
        self.outc = OutConv(in_channels=64, out_channels=out_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)
