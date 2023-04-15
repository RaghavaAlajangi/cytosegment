from pathlib import Path

import albumentations as A
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import from_numpy
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


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

    data_split = load_split_dataset(data_path, valid_size=valid_size)
    img_train, img_valid, msk_train, msk_valid = data_split

    train_dataset = UNetDataset(img_train, msk_train, augment=augmentation,
                                mean=mean, std=std)
    valid_dataset = UNetDataset(img_train, msk_train, augment=False,
                                mean=mean, std=std)

    data_dict = {
        "train": train_dataset,
        "valid": valid_dataset
    }

    dataloader_dict = create_dataloaders(data_dict, batch_size, num_workers)
    return dataloader_dict


def load_split_dataset(data_path, seed=42, valid_size=0.2):
    img_path = Path(data_path) / "images"
    msk_path = Path(data_path) / "masks"

    # Get the list of all the ".png" files
    img_list = [p for p in Path(img_path).rglob("*.png") if p.is_file()]
    msk_list = [p for p in Path(msk_path).rglob("*.png") if p.is_file()]

    # Read images and masks
    images = [np.asarray(Image.open(i)) for i in img_list]
    masks = [np.asarray(Image.open(i)) for i in msk_list]

    # Crop images and masks
    images = [img[8:72, :] for img in images]
    masks = [msk[8:72, :] for msk in masks]

    data_split = train_test_split(images, masks,
                                  test_size=valid_size,
                                  random_state=seed)
    return data_split


def create_dataloaders(data_dict, batch_size, num_workers=0, shuffle=False):
    dataloader_dict = dict()
    for m in data_dict.keys():
        dataloader_dict[m] = DataLoader(data_dict[m],
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=shuffle)
    return dataloader_dict


class UNetDataset(Dataset):
    def __init__(self, images, masks, augment=False,
                 normalize_mode="min_max", mean=None, std=None):
        self.images = images
        self.masks = masks
        self.augment = augment
        self.normalize_mode = normalize_mode
        self.mean = [0.] if mean is None else [mean]
        self.std = [1.] if std is None else [std]
        self.max_pix_val = 1.0 if normalize_mode == "min_max" else 255.

    @staticmethod
    def min_max_norm(img):
        return (img - img.min()) / (img.max() - img.min())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        msk = self.masks[index]

        if self.normalize_mode == "min_max":
            img = self.min_max_norm(img)

        msk = msk / msk.max()  # Make sure mask is binary
        if self.augment:
            compose_obj = A.Compose([
                # A.Rotate(7, border_mode=4, p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomBrightness(limit=(-1.0, 1.0), p=0.5),
                A.Normalize(self.mean, self.std,
                            max_pixel_value=self.max_pix_val)
            ])
        else:
            compose_obj = A.Compose([
                A.Normalize(self.mean, self.std,
                            max_pixel_value=self.max_pix_val)
            ])
        transformed = compose_obj(image=img, mask=msk)
        img_tensor = from_numpy(transformed["image"])
        mask_tensor = from_numpy(transformed["mask"])
        # Add an extra channel [H, W] --> [1, H, W]
        img_tensor = img_tensor.unsqueeze(0)
        # Apply padding to the images and masks (250 --> 256)
        padded_img = F.pad(img_tensor, (3, 3, 0, 0), value=0)
        padded_msk = F.pad(mask_tensor, (3, 3, 0, 0), value=0)
        return padded_img, padded_msk
