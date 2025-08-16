import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as tt

from .helper import read_data_files, split_data


def get_dataloaders(config):
    """Generates PyTorch DataLoader instances for training, validation, and
    testing datasets."""
    data_path = Path(config.data.path)

    if not data_path.is_dir():
        raise ValueError(
            f"The given path ({config.data.path}) is not a directory. Please "
            "provide a valid directory path that contains '/training' and "
            "'/testing' subdirectories."
        )

    train_images, train_masks = read_data_files(
        data_path / "training", seed=config.data.random_seed, shuffle=True
    )
    test_images, test_masks = read_data_files(
        data_path / "testing", seed=42, shuffle=False
    )

    train_images, valid_images, train_masks, valid_masks = split_data(
        train_images, train_masks, config.data.valid_size
    )

    datasets = {
        "train": UNetDataset(
            train_images,
            train_masks,
            img_size=config.data.img_size,
            augment=config.data.augmentation,
            mean=config.data.mean,
            std=config.data.std,
        ),
        "valid": UNetDataset(
            valid_images,
            valid_masks,
            img_size=config.data.img_size,
            augment=False,
            mean=config.data.mean,
            std=config.data.std,
        ),
        "test": UNetDataset(
            test_images,
            test_masks,
            img_size=config.data.img_size,
            augment=False,
            mean=config.data.mean,
            std=config.data.std,
        ),
    }

    dataloaders = {
        name: DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            pin_memory=True,
            shuffle=name == "train",
        )
        for name, dataset in datasets.items()
    }

    return dataloaders


class UNetDataset(Dataset):
    """Create torch dataset instance for training"""

    def __init__(
        self,
        image_paths,
        maks_paths,
        img_size,
        augment=False,
        min_max=False,
        mean=None,
        std=None,
    ):
        self.image_paths = image_paths
        self.mask_paths = maks_paths
        self.img_size = img_size
        self.augment = augment
        self.min_max = min_max
        self.mean = 0.0 if mean is None else mean
        self.std = 1.0 if std is None else std

    @staticmethod
    def min_max_norm(img):
        norm_np_img = (img - img.min()) / (img.max() - img.min())
        norm_ten_img = torch.tensor(norm_np_img, dtype=torch.float32)
        return norm_ten_img.unsqueeze(0)

    def resize_sample(self, image, mask, pad_value=0):
        height, width = image.shape
        target_height, target_width = self.img_size

        # don't do crop and padding if actual shape equal to target shape
        if (height, width) == (target_height, target_width):
            return image, mask

        # Calculate the difference in height and width
        hdiff = height - target_height
        wdiff = width - target_width

        hcorr = abs(hdiff // 2)
        wcorr = abs(wdiff // 2)

        # Adjust image height (crop or pad according to the target height)
        if hdiff > 0:
            # Cropping
            hcorr_img = image[hcorr : height - hcorr, :]  # noqa
            hcorr_msk = mask[hcorr : height - hcorr, :]  # noqa
        else:
            # Padding
            hcorr_img = np.full(
                (target_height, width), pad_value, dtype=np.float32
            )
            hcorr_msk = np.zeros((target_height, width), dtype=np.float32)
            hcorr_img[hcorr : hcorr + height, :] = image  # noqa
            hcorr_msk[hcorr : hcorr + height, :] = mask  # noqa

        # Adjust image width (crop or pad according to the target width)
        if wdiff > 0:
            # Cropping
            wcorr_img = hcorr_img[:, wcorr : width - wcorr]  # noqa
            wcorr_msk = hcorr_msk[:, wcorr : width - wcorr]  # noqa
        else:
            # Padding
            wcorr_img = np.full(
                (target_height, target_width), pad_value, dtype=np.float32
            )
            wcorr_msk = np.zeros(
                (target_height, target_width), dtype=np.float32
            )
            wcorr_img[:, wcorr : wcorr + width] = hcorr_img  # noqa
            wcorr_msk[:, wcorr : wcorr + width] = hcorr_msk  # noqa

        return wcorr_img, wcorr_msk

    def custom_transform(self, image, mask):
        # Instantiate normalize and to_tensor functions
        normalize = tt.Normalize([self.mean], [self.std])
        to_tensor = tt.ToTensor()

        # Make sure mask is binary and tensor
        mask = mask / mask.max()
        mask = torch.tensor(mask, dtype=torch.float32)

        if self.min_max:
            image = self.min_max_norm(image)
        else:
            # Normalize image (divides the image with 255)
            image = to_tensor(image.astype("uint8"))

        # Standardize image with mean and std values
        image = normalize(image)

        if self.augment:
            # Random horizontal flipping
            if random.random() >= 0.5:
                image = tf.hflip(image)
                mask = tf.hflip(mask)
            # Random vertical flipping
            if random.random() >= 0.5:
                image = tf.vflip(image)
                mask = tf.vflip(mask)

            # Apply brightness (add/subtract a random number from the image)
            if random.random() >= 0.5:
                brightness_factor = random.uniform(-1, 1)
                image = image + brightness_factor

        return image, mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path))

        # Compute image mean
        pad_value = image.mean()

        # Resize the image and mask sample according to the target shape
        resized_img, resized_msk = self.resize_sample(
            image, mask, pad_value=pad_value
        )
        # Augmentation
        aug_img, aug_msk = self.custom_transform(resized_img, resized_msk)

        return aug_img, aug_msk
