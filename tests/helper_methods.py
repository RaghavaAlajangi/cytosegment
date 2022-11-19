import pathlib

import torch


# import yaml


def get_data_dir():
    return pathlib.Path(__file__).parent / "data"


def get_test_tensors():
    # Create a test predict tensor
    predict = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                            1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
                            1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7])
    # Create a test target tensor
    target = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                           1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
                           1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7])
    # Convert test tensors into 2D tensors as if images and masks
    predict = predict.view(3, 1, 3, 3)
    target = target.view(3, 1, 3, 3)
    return predict, target
