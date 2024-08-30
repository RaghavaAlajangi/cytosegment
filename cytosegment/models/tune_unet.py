import torch
import torch.nn as nn
import torch.nn.functional as fun


class AttentionBlock(nn.Module):
    """Initializes the AttentionBlock"""

    def __init__(self, F_g, F_l, F_int, activation="relu"):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=(1, 1), stride=(1, 1), padding=0,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=(1, 1), stride=(1, 1), padding=0,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=(1, 1), stride=(1, 1), padding=0,
                      bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # Select the activation function
        self.activation = nn.ReLU(inplace=True) if activation == "relu" \
            else nn.LeakyReLU(0.1, inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class EncodeBlock(nn.Module):
    """Initializes the EncodeBlock"""

    def __init__(self, in_size, out_size, conv_block="double", dilation=1,
                 dropout=0, batch_norm=True, activation="relu"):
        super(EncodeBlock, self).__init__()
        # Create dilation kernel
        dilation_kernel = (dilation, dilation)
        activation_fuc = nn.ReLU(inplace=True) if activation == "relu" \
            else nn.LeakyReLU(0.1, inplace=True)

        block = [
            nn.Conv2d(in_size, out_size, kernel_size=(3, 3), padding=dilation,
                      dilation=dilation_kernel)
        ]

        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        block.append(activation_fuc)

        if conv_block == "double":
            block.append(nn.Conv2d(out_size, out_size, kernel_size=(3, 3),
                                   padding=dilation, dilation=dilation_kernel))
            if batch_norm:
                block.append(nn.BatchNorm2d(out_size))
            block.append(activation_fuc)

        if dropout > 0:
            block.append(nn.Dropout(p=dropout))

        self.block = nn.Sequential(*block)

    def forward(self, in_tensor):
        return self.block(in_tensor)


class DecodeBlock(nn.Module):
    """Initialize the DecodeBlock."""

    def __init__(self, in_size, out_size, up_mode="upconv",
                 conv_block="double", dilation=1, dropout=0, batch_norm=True,
                 activation="relu", attention=False):
        super(DecodeBlock, self).__init__()
        self.attention = attention
        self.conv_block = EncodeBlock(in_size, out_size, conv_block, dilation,
                                      dropout, batch_norm, activation)

        self.attn_block = AttentionBlock(F_g=out_size, F_l=out_size,
                                         F_int=out_size // 2,
                                         activation=activation)

        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=(2, 2),
                                         stride=(2, 2))
        elif up_mode == "upsample":
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2,
                            align_corners=True),
                nn.Conv2d(in_size, out_size, kernel_size=(1, 1)),
            )

    def forward(self, prev_encode, curr_decode):
        up_prev_encode = self.up(prev_encode)
        if self.attention:
            atten_out = self.attn_block(up_prev_encode, curr_decode)
            cat_out = torch.cat((atten_out, up_prev_encode), dim=1)
            out = self.conv_block(cat_out)
        else:
            x = torch.cat([curr_decode, up_prev_encode], dim=1)
            out = self.conv_block(x)
        return out


class TunableUNet(nn.Module):
    """Tunable U-Net for image segmentation with configurable depth,
    filters, and other hyperparameters.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_classes : int
        Number of output classes.
    conv_block : str
        Type of convolutional block, one of {"single", "double"}. Defaults
        to "double".
    depth : int
        Depth of the network. Defaults to 3.
    filters : int
        Base number of filters. Defaults to 3
    dilation : int
        Dilation rate for convolutional layers. Defaults to 1.
        1 means no dilation.
    dropout : float
        Dropout rate. Defaults to 0.
    up_mode : str
        Upsampling mode, one of {"upconv", "upsample"}. Defaults to "upconv".
    batch_norm : bool
        Whether to use batch normalization. Defaults to True.
    attention : bool
        Whether to use attention blocks. Defaults to False.
    activation : str
        Activation function, one of {"relu", "leaky_relu"}. Defaults to "relu".
    """

    def __init__(self, **kwargs):
        super(TunableUNet, self).__init__()
        # Set default parameters
        params = {
            "in_channels": 1,
            "out_classes": 1,
            "conv_block": "double",
            "depth": 3,
            "filters": 3,
            "dilation": 1,
            "dropout": 0,
            "up_mode": "upconv",
            "batch_norm": True,
            "attention": False,
            "activation": "relu"
        }
        # Update parameters with provided arguments
        params.update(kwargs)

        prev_channels = params["in_channels"]

        # Create encoding path
        self.encoder = nn.ModuleList()
        for i in range(params["depth"] + 1):
            out_channels = 2 ** (params["filters"] + i)
            self.encoder.append(
                EncodeBlock(prev_channels, out_channels, params["conv_block"],
                            params["dilation"], params["dropout"],
                            params["batch_norm"], params["activation"])
            )
            prev_channels = out_channels

        # Create decoding path
        self.decoder = nn.ModuleList()
        for i in reversed(range(params["depth"])):
            out_channels = 2 ** (params["filters"] + i)
            self.decoder.append(
                DecodeBlock(prev_channels, out_channels, params["up_mode"],
                            params["conv_block"], params["dilation"],
                            params["dropout"], params["batch_norm"],
                            params["activation"], params["attention"])
            )
            prev_channels = out_channels

        # Final convolutional layer
        self.last = nn.Conv2d(prev_channels, params["out_classes"],
                              kernel_size=(1, 1))

    def forward(self, in_tensor):
        blocks = []
        # Iterate through the encoder blocks
        for idx, encode in enumerate(self.encoder):
            # Pass the input through the current encoder block
            in_tensor = encode(in_tensor)

            # If not last encoder block, store the output and down-sample it.
            if idx != len(self.encoder) - 1:
                blocks.append(in_tensor)
                in_tensor = fun.max_pool2d(in_tensor, 2)

        # Iterate through the decoder blocks in reverse order
        for idx, decode in enumerate(self.decoder):
            # Pass the input through the current decoder block with the
            # corresponding feature map
            in_tensor = decode(in_tensor, blocks[-idx - 1])

        # Pass the output through the final convolutional layer
        logits = self.last(in_tensor)

        # Apply sigmoid activation to the output
        return torch.sigmoid(logits)
