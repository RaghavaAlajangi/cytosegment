from pathlib import Path

import albumentations as A
import h5py
import torch
from torch import from_numpy
from torch.utils.data import Dataset, DataLoader, random_split

from .labelme_utils import json_to_mask


def get_dataloaders_with_params(params):
    assert {"dataset"}.issubset(params)
    dataset_params = params.get("dataset")
    assert {"type"}.issubset(dataset_params)
    data_type = dataset_params.get("type")

    assert {"data_path", "augmentation"}.issubset(dataset_params)
    assert {"valid_size", "batch_size"}.issubset(dataset_params)
    assert {"mean", "std", "num_workers"}.issubset(dataset_params)
    assert {"num_samples"}.issubset(dataset_params)

    data_path = dataset_params.get("data_path")
    augmentation = dataset_params.get("augmentation")
    valid_size = dataset_params.get("valid_size")
    batch_size = dataset_params.get("batch_size")
    mean = dataset_params.get("mean")
    std = dataset_params.get("std")
    num_workers = dataset_params.get("num_workers")
    num_samples = dataset_params.get("num_samples")

    if data_type.lower() == "json":
        unet_dataset = UNetDataset.from_json_files(data_path, augmentation,
                                                   mean, std, num_samples)
    else:
        unet_dataset = UNetDataset.from_hdf5_data(data_path, augmentation,
                                                  mean, std, num_samples)

    data_dict = split_dataset(unet_dataset, valid_size)
    dataloader_dict = create_dataloaders(data_dict, batch_size, num_workers)
    return dataloader_dict


class UNetDataset(Dataset):
    def __init__(self, images, masks,
                 augment=False, mean=None, std=None):
        self.images = images
        self.masks = masks
        self.augment = augment
        self.mean = [0.] if mean is None else mean
        self.std = [1.] if std is None else std

    @classmethod
    def from_hdf5_data(cls, hdf5_file, augment=False, mean=None,
                       std=None, num_samples=None):
        hdf5_ds = h5py.File(hdf5_file)["events"]
        ds_len = len(hdf5_ds["image"])
        num_samples = num_samples if num_samples else ds_len
        images = hdf5_ds["image"][:num_samples]
        masks = hdf5_ds["mask"][:num_samples]
        return cls(images, masks, augment, mean, std)

    @classmethod
    def from_json_files(cls, json_path, augment=False, mean=None,
                        std=None, num_samples=None):
        json_list = [p for p in Path(json_path).rglob("*.json") if p.is_file()]
        images, masks = json_to_mask(json_list)

        len_images = len(images)
        num_samples = num_samples if num_samples else len_images
        images = images[:num_samples]
        masks = masks[:num_samples]
        return cls(images, masks, augment, mean, std)

    @staticmethod
    def min_max_norm(img):
        return (img - img.min()) / (img.max() - img.min())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        msk = self.masks[index]
        img = self.min_max_norm(img)
        msk = msk / msk.max()  # Make sure mask is binary
        if self.augment:
            compose_obj = A.Compose([
                A.Rotate(7, border_mode=4, p=0.5),
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.3),
                A.Normalize(self.mean, self.std, max_pixel_value=1.0)
            ])
        else:
            compose_obj = A.Compose([
                A.Normalize(self.mean, self.std, max_pixel_value=1.0)
            ])
        transformed = compose_obj(image=img, mask=msk)
        img_tensor = from_numpy(transformed["image"])
        mask_tensor = from_numpy(transformed["mask"])
        # Add an extra channel [H, W] --> [1, H, W]
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor, mask_tensor


def split_dataset(dataset_object, valid_size=0.2):
    len_dataset = dataset_object.__len__()
    valid_data_size = int(valid_size * len_dataset)
    train_data_size = int(len_dataset - valid_data_size)
    train_data, valid_data = random_split(dataset_object,
                                          [train_data_size, valid_data_size])
    data = {"train": train_data, "valid": valid_data}
    return data


def create_dataloaders(data_dict, batch_size, num_workers=0):
    data_load_dict = dict()
    for m in data_dict.keys():
        data_load_dict[m] = DataLoader(data_dict[m],
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       shuffle=True)
    return data_load_dict


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
