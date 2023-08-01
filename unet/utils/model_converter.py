from pathlib import Path

import torch

from unet.ml_models import UNetTunable


def convert_torch_to_onnx(torch_ckp_path):
    model = UNetTunable(in_channels=1,
                        out_classes=1,
                        conv_block="double",
                        depth=4,
                        filters=3,
                        dilation=1,
                        dropout=0,
                        up_mode="upconv",
                        batch_norm=True,
                        attention=False,
                        relu=True)
    # model = DataParallel(model)
    cuda_device = torch.device('cpu')
    ckp = torch.load(torch_ckp_path, map_location=cuda_device)
    model.load_state_dict(ckp['model_state_dict'])
    model.eval()

    onnx_path = Path(torch_ckp_path).with_suffix(".onnx")

    batch_size = 8
    dummy_input = torch.randn(batch_size, 1, 64, 256, requires_grad=True)
    dummy_input = dummy_input.to(cuda_device, dtype=torch.float32)

    # Export the model
    torch.onnx.export(model,  # model being run
                      # model input (or a tuple for multiple inputs)
                      dummy_input,
                      # where to save the model (can be a file-like object)
                      str(onnx_path),
                      # store the trained  weights inside the model file
                      export_params=True,
                      # the ONNX version to export the model to
                      opset_version=11,
                      # whether to execute constant folding for optimization
                      do_constant_folding=True,
                      # the model's input names
                      input_names=['input'],
                      # the model's output names
                      output_names=['output'],
                      # variable length axes
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})


if __name__ == "__main__":
    p = "path/to/torch/model"
    convert_torch_to_onnx(p)
