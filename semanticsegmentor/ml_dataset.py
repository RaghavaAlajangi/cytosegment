from pathlib import Path
import random
import zipfile

import numpy as np
from PIL import Image
import torch
from torch import mean as tmean
from torch.utils.data import Dataset, DataLoader


def get_dataloaders_with_params(params):
    assert {"dataset"}.issubset(params)
    dataset_params = params.get("dataset")
    assert {"type"}.issubset(dataset_params)
    # data_type = dataset_params.get("type")

    assert {"data_path", "augmentation"}.issubset(dataset_params)
    assert {"valid_size", "batch_size"}.issubset(dataset_params)
    assert {"mean", "std", "num_workers"}.issubset(dataset_params)
    assert {"min_max", "img_size"}.issubset(dataset_params)
    assert {"random_seed"}.issubset(dataset_params)

    data_path = dataset_params.get("data_path")
    augmentation = dataset_params.get("augmentation")
    valid_size = dataset_params.get("valid_size")
    batch_size = dataset_params.get("batch_size")
    img_size = dataset_params.get("img_size")
    mean = dataset_params.get("mean")
    std = dataset_params.get("std")
    num_workers = dataset_params.get("num_workers")
    min_max = dataset_params.get("min_max")
    random_seed = dataset_params.get("random_seed")

    train_data_path, test_data_path = unzip_data(data_path)

    images, masks = process_data(train_data_path, img_size, seed=random_seed,
                                 shuffle=True)

    train_imgs, valid_imgs, train_msks, valid_msks = split_data(images, masks,
                                                                valid_size)

    test_imgs, test_msks = process_data(test_data_path, img_size, seed=42,
                                        shuffle=False)

    # Create training dataset instance
    train_dataset = UNetDataset(train_imgs, train_msks, augment=augmentation,
                                min_max=min_max, mean=mean, std=std)
    # Create validation dataset instance and make sure augmentation is False
    valid_dataset = UNetDataset(valid_imgs, valid_msks, augment=False,
                                min_max=min_max, mean=mean, std=std)
    # Create testing dataset instance
    test_dataset = UNetDataset(test_imgs, test_msks, augment=False,
                               min_max=min_max, mean=mean, std=std)

    data_dict = {
        "train": train_dataset,
        "valid": valid_dataset,
        "test": test_dataset,
    }
    # Training data will be shuffled
    dataloader_dict = create_dataloaders(data_dict, batch_size, num_workers)
    return dataloader_dict


def unzip_data(zipped_data_path):
    """Unzip data path and return train and test data paths"""

    # Create output path from input path
    pathout = Path(zipped_data_path).with_suffix("")

    # Create train and test datasets output paths
    train_data_path = pathout / "training"
    test_data_path = pathout / "testing"

    # Extract zipped file, if train and test dirs are not existed.
    if not train_data_path.exists() or not test_data_path.exists():
        with zipfile.ZipFile(zipped_data_path, "r") as zip_ref:
            zip_ref.extractall(pathout.parents[0])

    return train_data_path, test_data_path


def process_data(data_path, img_size, seed=42, shuffle=False):
    img_path = Path(data_path) / "images"
    msk_path = Path(data_path) / "masks"

    # Get the list of all the ".png" files
    img_list = sorted([p for p in Path(img_path).rglob("*.png")
                       if p.is_file()])
    msk_list = sorted([p for p in Path(msk_path).rglob("*.png")
                       if p.is_file()])

    assert len(img_list) == len(msk_list)

    if shuffle:
        # Shuffle img_list and msk_list
        np.random.seed(seed)
        np.random.shuffle(img_list)
        np.random.seed(seed)
        np.random.shuffle(msk_list)

    images = []
    masks = []

    for img_path, msk_path in zip(img_list, msk_list):
        img = np.array(Image.open(img_path))
        msk = np.array(Image.open(msk_path))
        img, msk = crop_pad_data(img, msk, img_size)
        images.append(img)
        masks.append(msk)

    return images, masks


def crop_pad_data(image, mask, img_size):
    height, width = image.shape
    target_height, target_width = img_size
    im_mean = image.mean()

    height_diff = height - target_height
    width_diff = width - target_width

    if height_diff > 0:
        height_corr = abs(height_diff) // 2
        new_img = image[height_corr: height - height_corr, :]
        new_msk = mask[height_corr: height - height_corr, :]
    else:
        hpad = abs(height_diff) // 2
        new_img = np.pad(image, ((hpad, hpad), (0, 0)),
                         constant_values=(im_mean, im_mean))
        new_msk = np.pad(mask, ((hpad, hpad), (0, 0)), constant_values=(0, 0))

    if width_diff > 0:
        width_corr = abs(width_diff) // 2
        new_img = new_img[:, width_corr: width - width_corr]
        new_msk = new_msk[:, width_corr: width - width_corr]
    else:
        wpad = abs(width_diff) // 2
        new_img = np.pad(new_img, ((0, 0), (wpad, wpad)),
                         constant_values=(im_mean, im_mean))
        new_msk = np.pad(new_msk, ((0, 0), (wpad, wpad)),
                         constant_values=(0, 0))
    return new_img, new_msk


def split_data(images, masks, valid_size=0.2):
    assert len(images) == len(masks)
    # Get the length of the dataset
    len_dataset = len(images)
    # Compute the train samples
    train_img_samples = int((1 - valid_size) * len_dataset)
    # Slicing train and valid samples
    train_imgs = images[:train_img_samples]
    test_imgs = images[train_img_samples:]

    train_msks = masks[:train_img_samples]
    test_msks = masks[train_img_samples:]

    return train_imgs, test_imgs, train_msks, test_msks


def create_dataloaders(data_dict, batch_size, num_workers=0):
    dataloader_dict = dict()
    for k in data_dict.keys():
        dataloader_dict[k] = DataLoader(
            data_dict[k],
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True if k == "train" else False
        )
    return dataloader_dict


def compute_mean_std(data_path, img_size, min_max=False):
    """ Computes the mean and standard deviation of a dataset.
    Parameters
    ----------
    data_path: str or Path
        Data directory path that has images and masks directories
    img_size: tuple
        Desired image size. Image samples are padded or cropped according
        to the img_size automatically
    min_max: bool
        Determine whether to use the min and max

    Returns
    -------
    The mean and standard deviation of the training data
    """
    images, masks = process_data(data_path, img_size, seed=42, shuffle=False)
    data_dict = {"data": UNetDataset(images, masks, augment=False,
                                     min_max=min_max)
                 }

    # Training data will be shuffled
    dataloader_dict = create_dataloaders(data_dict, batch_size=8)

    channel_sum, channel_square_sum, batch_counter = 0, 0, 0

    for imgs, _ in dataloader_dict["data"]:
        channel_sum += tmean(imgs, dim=[0, 2, 3])
        channel_square_sum += tmean(imgs ** 2, dim=[0, 2, 3])

        batch_counter += 1

    mean = float(channel_sum / batch_counter)
    std = float((channel_square_sum / batch_counter - mean ** 2) ** 0.5)
    return mean, std


class UNetDataset(Dataset):
    """ Create torch dataset instance for training"""

    def __init__(self, images, masks, augment=False,
                 min_max=False, mean=None, std=None):
        self.images = images
        self.masks = masks
        self.min_max = min_max
        self.augment = augment
        self.mean = 0. if mean is None else mean
        self.std = 1. if std is None else std

    @staticmethod
    def min_max_norm(img):
        norm_np_img = (img - img.min()) / (img.max() - img.min())
        norm_ten_img = torch.tensor(norm_np_img, dtype=torch.float32)
        return norm_ten_img.unsqueeze(0)

    def custom_transform(self, image, mask):
        # Normalize image and mask (mapping to [0, 1])
        mask = mask / 255.0
        image = image / 255.0

        # Standardize image
        image = (image - self.mean) / self.std

        # Expand image dimension [H, W] -> [1, 1, H, W]
        exp_img = np.expand_dims(image, axis=(0, 1))
        # Expand mask dimension [H, W] -> [1, H, W]
        exp_msk = np.expand_dims(mask, axis=0)

        if self.augment:
            image_copy = exp_img.copy()
            mask_copy = exp_msk.copy()

            # Apply horizontal (left to right) flipping
            image_flip = np.flip(image_copy, axis=-1)
            mask_flip = np.flip(mask_copy, axis=-1)

            # Apply brightness
            # add a random number in between (-1, 1) from the image)
            brightness_factor = random.uniform(-1, 1)
            image_flip = image_flip + brightness_factor

            # Concatenate original and augmented samples
            trans_img = np.concatenate((image_copy, image_flip), axis=0)
            trans_msk = np.concatenate((mask_copy, mask_flip), axis=0)

            # Convert numpy to float tensor
            img_ten = torch.tensor(trans_img, dtype=torch.float32)
            msk_ten = torch.tensor(trans_msk, dtype=torch.float32)

            return img_ten, msk_ten

        else:
            # Convert numpy to float tensor
            img_ten = torch.tensor(exp_img, dtype=torch.float32)
            msk_ten = torch.tensor(exp_msk, dtype=torch.float32)

            return img_ten, msk_ten

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        msk = self.masks[index]
        # Apply transforms
        img_tensor, mask_tensor = self.custom_transform(img, msk)
        return img_tensor, mask_tensor
