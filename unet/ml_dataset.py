from pathlib import Path
import random
import zipfile

import albumentations as A
import h5py
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tt
import torchvision.transforms.functional as tf


def get_dataloaders_with_params(params):
    assert {"dataset"}.issubset(params)
    dataset_params = params.get("dataset")
    assert {"type"}.issubset(dataset_params)
    data_type = dataset_params.get("type")

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

    pathout = Path(data_path)
    unzip_data(data_path, pathout.parents[0])

    train_data_path = pathout.with_suffix('') / "training"

    img_files, msk_files = get_data_files(train_data_path, seed=random_seed,
                                          shuffle=True)
    images, masks = read_data(img_files, msk_files)
    resized_data = [crop_pad_data(img, msk, img_size) for img, msk in
                    zip(images, masks)]

    images = [sublist[0] for sublist in resized_data]
    masks = [sublist[1] for sublist in resized_data]

    train_imgs, valid_imgs, train_msks, valid_msks = split_data(images, masks,
                                                                valid_size)

    train_dataset = UNetDataset(train_imgs, train_msks, augment=augmentation,
                                min_max=min_max,
                                mean=mean, std=std)
    # Make sure augmentation is False for validation dataset
    valid_dataset = UNetDataset(valid_imgs, valid_msks, augment=False,
                                min_max=min_max,
                                mean=mean, std=std)

    data_dict = {
        "train": train_dataset,
        "valid": valid_dataset
    }
    # Training data will be shuffled
    dataloader_dict = create_dataloaders(data_dict, batch_size, num_workers)
    return dataloader_dict


def unzip_data(pathin, pathout):
    with zipfile.ZipFile(pathin, 'r') as zip_ref:
        zip_ref.extractall(pathout)


def get_data_files(data_path, seed=42, shuffle=False):
    img_path = Path(data_path) / "images"
    msk_path = Path(data_path) / "masks"

    # Get the list of all the ".png" files
    img_list = sorted([p for p in Path(img_path).rglob("*.png") if p.is_file()])
    msk_list = sorted([p for p in Path(msk_path).rglob("*.png") if p.is_file()])

    assert len(img_list) == len(msk_list)

    if shuffle:
        # Shuffle img_list and msk_list
        np.random.seed(seed)
        np.random.shuffle(img_list)
        np.random.seed(seed)
        np.random.shuffle(msk_list)
    return img_list, msk_list


def read_data(img_list, msk_list):
    # Read images and masks
    images = [np.array(Image.open(i)) for i in img_list]
    masks = [np.array(Image.open(i)) for i in msk_list]
    return images, masks


def crop_pad_data(image, mask, img_size):
    height, width = image.shape
    target_height, target_width = img_size

    height_diff = height - target_height
    width_diff = width - target_width

    if height_diff > 0:
        height_corr = abs(height_diff) // 2
        new_img = image[height_corr: height - height_corr, :]
        new_msk = mask[height_corr: height - height_corr, :]
    else:
        hpad = abs(height_diff) // 2
        new_img = np.pad(image, ((hpad, hpad), (0, 0)), mode="mean")
        new_msk = np.pad(mask, ((hpad, hpad), (0, 0)), mode="constant")

    if width_diff > 0:
        width_corr = abs(width_diff) // 2
        new_img = new_img[:, width_corr: width - width_corr]
        new_msk = new_msk[:, width_corr: width - width_corr]
    else:
        wpad = abs(width_diff) // 2
        new_img = np.pad(new_img, ((0, 0), (wpad, wpad)), mode="mean")
        new_msk = np.pad(new_msk, ((0, 0), (wpad, wpad)), mode="constant")
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


class UNetDataset(Dataset):
    def __init__(self, images, masks, augment=False,
                 min_max=False, mean=None, std=None):
        self.images = images
        self.masks = masks
        self.min_max = min_max
        self.augment = augment
        self.mean = [0.] if mean is None else [mean]
        self.std = [1.] if std is None else [std]

    @staticmethod
    def min_max_norm(img):
        norm_np_img = (img - img.min()) / (img.max() - img.min())
        norm_ten_img = torch.tensor(norm_np_img, dtype=torch.float32)
        return norm_ten_img.unsqueeze(0)

    def custom_transform(self, image, mask):
        # Instantiate normalize and to_tensor functions
        normalize = tt.Normalize(self.mean, self.std)
        to_tensor = tt.ToTensor()

        # Make sure mask is binary and tensor
        mask = mask / mask.max()
        mask = torch.tensor(mask, dtype=torch.float32)

        if self.min_max:
            image = self.min_max_norm(image)
        else:
            # Normalize image (divides the image with 255)
            image = to_tensor(image)

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
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        msk = self.masks[index]
        # Apply transforms
        img_tensor, mask_tensor = self.custom_transform(img, msk)
        return img_tensor, mask_tensor


class RTDCInference(Dataset):
    def __init__(self, rtdc_path, normalize_mode="max_pixel",
                 mean=None, std=None):
        rtdc_data = h5py.File(rtdc_path)["events"]
        self.images = rtdc_data["image"]
        self.normalize_mode = normalize_mode,
        mean = [0.] if mean is None else [mean]
        std = [1.] if std is None else [std]
        max_pix_val = 1.0 if normalize_mode == "min_max" else 255.

        self.transform = A.Compose([
            A.Normalize(mean, std, max_pixel_value=max_pix_val)
        ])

    @staticmethod
    def min_max_norm(img):
        return (img - img.min()) / (img.max() - img.min())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        img = img[8:72, :]

        if self.normalize_mode == "min_max":
            img = self.min_max_norm(img)

        transformed = self.transform(image=img)
        img_tensor = torch.from_numpy(transformed["image"])
        # Add an extra channel [H, W] --> [1, H, W]
        img_tensor = img_tensor.unsqueeze(0)
        # Apply padding to the images and masks (250 --> 256)
        padded_img = F.pad(img_tensor, (3, 3, 0, 0), value=0)
        return padded_img
