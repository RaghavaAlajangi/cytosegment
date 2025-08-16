import torch
from torch.utils.data import DataLoader

from .helper import read_data_files
from .manager import UNetDataset


def compute_data_mean_std(data_path, img_size):
    """Computes the mean and standard deviation of a dataset.

    Parameters
    ----------
    data_path: str or Path
        Data directory path that has images and masks directories
    img_size: tuple
        Desired image size. Image samples are padded or cropped according
        to the img_size automatically

    Returns
    -------
    The mean and standard deviation of the training data
    """
    image_files, mask_files = read_data_files(
        data_path, seed=42, shuffle=False
    )
    dataset = UNetDataset(
        image_files, mask_files, target_shape=img_size, augment=False
    )

    dataloader = DataLoader(dataset, batch_size=32, pin_memory=True)

    mean = 0
    std = 0
    num_samples = 0

    for images, _ in dataloader:
        batch_mean = torch.mean(images, dim=[0, 2, 3])
        batch_std = torch.std(images, dim=[0, 2, 3])

        num_samples += images.shape[0]
        mean = (
            mean * (num_samples - images.shape[0])
            + batch_mean * images.shape[0]
        ) / num_samples
        std = (
            std * (num_samples - images.shape[0]) + batch_std * images.shape[0]
        ) / num_samples

    return mean.item(), std.item()
