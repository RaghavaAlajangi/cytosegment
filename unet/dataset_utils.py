import torch
from torch.utils.data import DataLoader

from .dataset import UNetDataset


def compute_mean_std(hdf5_file_path):
    dataset = UNetDataset.from_hdf5_data(hdf5_file_path)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    channel_sum, channel_square_sum, batch_counter = 0, 0, 0

    for imgs, _ in data_loader:
        channel_sum += torch.mean(imgs, dim=[0, 2, 3])
        channel_square_sum += torch.mean(imgs ** 2, dim=[0, 2, 3])

        batch_counter += 1

    mean = float(channel_sum / batch_counter)
    std = float((channel_square_sum / batch_counter - mean ** 2) ** 0.5)
    return mean, std
