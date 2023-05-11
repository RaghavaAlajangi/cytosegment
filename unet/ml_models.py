import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def get_model_with_params(params):
    assert {"model"}.issubset(params)
    model_params = params.get("model")
    assert {"type"}.issubset(model_params)
    model_type = model_params.get("type")
    if model_type.lower() == "unet":
        assert {"in_channels", "out_classes"}.issubset(model_params)
        assert {"bilinear"}.issubset(model_params)
        return UNet(in_channels=model_params.get("in_channels"),
                    out_classes=model_params.get("out_classes"),
                    bilinear=model_params.get("bilinear"))

    if model_type.lower() == "unettunable":
        assert {"in_channels", "out_classes"}.issubset(model_params)
        assert {"depth", "filters"}.issubset(model_params)
        assert {"dilation", "dropout"}.issubset(model_params)
        assert {"up_mode", "attention"}.issubset(model_params)
        assert {"relu"}.issubset(model_params)
        return UNetTunable(in_channels=model_params.get("in_channels"),
                           out_classes=model_params.get("out_classes"),
                           depth=model_params.get("depth"),
                           filters=model_params.get("filters"),
                           dilation=model_params.get("dilation"),
                           dropout=model_params.get("dropout"),
                           up_mode=model_params.get("up_mode"),
                           attention=model_params.get("attention"),
                           relu=model_params.get("relu"))


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3),
                      padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3),
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of
        # channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear",
                                  align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                         kernel_size=(3, 3), stride=(2, 2))
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # if you have padding issues, see
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Find the code here:
    https://github.com/xiaopeng-liao/Pytorch-UNet
    """

    def __init__(self, in_channels, out_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_classes)

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
        return logits


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int, relu=True):
        super(AttentionBlock, self).__init__()

        # init.normal_(self.conv.weight, mean=0, std=0.01)

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=(1, 1), stride=(1, 1),
                      padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=(1, 1), stride=(2, 2),
                      padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=(1, 1), stride=(1, 1),
                      padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # Select the activation function
        if relu:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(0.1, inplace=True)

        self.up = nn.Upsample(mode="bilinear", scale_factor=2,
                              align_corners=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)
        up_samp = self.up(psi)
        return x * up_samp


class EncodingBlock(nn.Module):
    def __init__(self, in_size, out_size, dilation=1, dropout=0,
                 relu=True):
        super(EncodingBlock, self).__init__()

        # Create dilation kernel and padding based on dilation argument
        dilation_kernel = (1, 1) if dilation == 1 else (dilation, dilation)
        # Select the activation function
        if relu:
            activation = nn.ReLU(inplace=True)
        else:
            activation = nn.LeakyReLU(0.1, inplace=True)

        block = [
            nn.Conv2d(in_size, out_size,
                      kernel_size=(3, 3),
                      padding=dilation,
                      dilation=dilation_kernel),
            nn.BatchNorm2d(out_size),
            activation,
            nn.Conv2d(out_size, out_size,
                      kernel_size=(3, 3),
                      padding=dilation,
                      dilation=dilation_kernel),
            nn.BatchNorm2d(out_size),
            activation,
        ]
        if dropout > 0:
            block.append(nn.Dropout(p=dropout))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class DecodingBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, attention=False,
                 relu=True):
        self.attention = attention
        super(DecodingBlock, self).__init__()
        self.conv_block = EncodingBlock(in_size, out_size,
                                        relu=relu)

        self.attn_block = AttentionBlock(F_g=in_size, F_l=out_size,
                                         F_int=out_size, relu=relu)

        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(in_size, out_size,
                                         kernel_size=(2, 2),
                                         stride=(2, 2))
        elif up_mode == "upsample":
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2,
                            align_corners=True),
                nn.Conv2d(in_size, out_size, kernel_size=(1, 1)),
            )

    def forward(self, prev_encode, curr_decode):
        if self.attention:
            out = self.attn_block(prev_encode, curr_decode)
            return out
        else:
            # Decode the final layer of encoding block
            up_prev_encode = self.up(prev_encode)
            # Concat up and skip-connection layers
            x = torch.cat([curr_decode, up_prev_encode], dim=1)
            out = self.conv_block(x)
            return out


class UNetTunable(nn.Module):
    def __init__(self, in_channels=1, out_classes=1, depth=5, filters=6,
                 dilation=1, dropout=0, up_mode="upconv", attention=False,
                 relu=True):
        """
        Implementation of U-Net: Convolutional Networks for Biomedical
        Image Segmentation (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper

        Parameters
        ----------
            in_channels: int
                Number of input channels
            out_classes: int
                Number of output channels
            depth: int
                Depth of the network
            filters: int
                Number of filters in the CNN layers is created based on
                "filters" and "depth" arguments in a loop by using the
                below formula.
                for i in range(depth):
                    2**(filters + i)
            dilation: int
                Dilation rate to be introduced
            up_mode: str
                one of "upconv" or "upsample"."upconv" uses transposed
                convolutions for learned upsampling. "upsample" will use
                bilinear upsampling.
            with_attn: boolean
                specify whether attention blocks are included when building
                the model

        Returns
        -------
        """

        super(UNetTunable, self).__init__()
        assert up_mode in ("upconv", "upsample")

        prev_channels = in_channels

        # Create encoding path
        self.encoder = nn.ModuleList()
        for i in range(depth):
            out_channels = 2 ** (filters + i)
            self.encoder.append(
                EncodingBlock(prev_channels, out_channels,
                              dilation=dilation, dropout=dropout,
                              relu=relu)
            )
            prev_channels = out_channels

        # Creating decoding path
        self.decoder = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            out_channels = 2 ** (filters + i)
            self.decoder.append(
                DecodingBlock(prev_channels, out_channels, up_mode,
                              attention=attention, relu=relu)
            )
            prev_channels = out_channels

        self.last = nn.Conv2d(prev_channels, out_classes, kernel_size=(1, 1))

    def forward(self, x):
        blocks = []
        for i, encode in enumerate(self.encoder):
            x = encode(x)
            if i != len(self.encoder) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, decode in enumerate(self.decoder):
            x = decode(x, blocks[-i - 1])

        return self.last(x)
