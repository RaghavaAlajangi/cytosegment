from .bench_unet import BenchmarkUNet
from .tune_unet import TunableUNet
from .utils import (add_params_to_jit_model, convert_torch_to_onnx,
                    init_weights, summary)


def get_model(config):
    model_name = config.model.name.lower()
    weight_init = config.model.weight_init.lower()

    model_params = {
        "in_channels": config.model.in_channels,
        "out_classes": config.model.out_classes
    }

    if model_name == "tunableunet":
        model_params.update({
            "conv_block": config.model.conv_block,
            "depth": config.model.depth,
            "filters": config.model.filters,
            "dilation": config.model.dilation,
            "dropout": config.model.dropout,
            "batch_norm": config.model.batch_norm,
            "up_mode": config.model.up_mode,
            "attention": config.model.attention,
            "relu": config.model.relu
        })
        model = TunableUNet(**model_params)
    else:
        model_params.update({"bilinear": True})
        model = BenchmarkUNet(**model_params)

    if weight_init != "default":
        return init_weights(model, init_type=weight_init)
    return model
