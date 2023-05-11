from pathlib import Path
import zipfile

import albumentations as A
import h5py
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import from_numpy
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
    assert {"min_max"}.issubset(dataset_params)

    data_path = dataset_params.get("data_path")
    augmentation = dataset_params.get("augmentation")
    valid_size = dataset_params.get("valid_size")
    batch_size = dataset_params.get("batch_size")
    mean = dataset_params.get("mean")
    std = dataset_params.get("std")
    num_workers = dataset_params.get("num_workers")
    min_max = dataset_params.get("min_max")

    pathout = Path(data_path).parents[0]
    unzip_data(data_path, pathout)

    train_data_path = pathout / "training_testing_set_w_beads" / "training"

    images, masks = read_suffle_data(train_data_path)
    images, masks = crop_data(images, masks)
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


def read_suffle_data(data_path, seed=42):
    img_path = Path(data_path) / "images"
    msk_path = Path(data_path) / "masks"

    # Get the list of all the ".png" files
    img_list = [p for p in Path(img_path).rglob("*.png") if p.is_file()]
    msk_list = [p for p in Path(msk_path).rglob("*.png") if p.is_file()]

    # Read images and masks
    images = [np.asarray(Image.open(i)) for i in img_list]
    masks = [np.asarray(Image.open(i)) for i in msk_list]

    # Shuffle images and masks
    np.random.seed(seed)
    np.random.shuffle(images)
    np.random.seed(seed)
    np.random.shuffle(masks)
    return images, masks


def crop_data(images, masks):
    images = [img[8:72, :] for img in images]
    masks = [msk[8:72, :] for msk in masks]
    return images, masks


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
        dataloader_dict[k] = DataLoader(data_dict[k],
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        pin_memory=True,
                                        shuffle=True if k == "train" else False)
    return dataloader_dict


class UNetDataset(Dataset):
    def __init__(self, images, masks, augment=False,
                 min_max=False, mean=None, std=None):
        self.images = images
        self.masks = masks
        self.augment = augment
        self.min_max = min_max
        self.mean = [0.] if mean is None else [mean]
        self.std = [1.] if std is None else [std]
        self.max_pix_val = 1.0 if min_max else 255.

    @staticmethod
    def min_max_norm(img):
        return (img - img.min()) / (img.max() - img.min())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        msk = self.masks[index]

        # img = np.pad(img, [(0, 0), (3, 3)], mode='edge')
        # msk = np.pad(msk, [(0, 0), (3, 3)], mode='edge')

        if self.min_max:
            img = self.min_max_norm(img)

        msk = msk / msk.max()  # Make sure mask is binary
        if self.augment:
            compose_obj = A.Compose([
                # A.Rotate(3, p=0.5),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                # A.RandomBrightness(limit=(-1.0, 1.0), p=0.5),
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
        mask_tensor = mask_tensor.unsqueeze(0)
        # Apply padding to the images and masks (250 --> 256)
        img_tensor = F.pad(img_tensor, (3, 3, 0, 0), value=0)
        mask_tensor = F.pad(mask_tensor, (3, 3, 0, 0), value=0)
        # return padded_img, padded_msk
        return img_tensor, mask_tensor

    #

    # class UNetDataset(Dataset):
    #     def __init__(self, images, masks, augment=False,
    #                  min_max=False, mean=None, std=None):
    #         self.images = images
    #         self.masks = masks
    #         self.augment = augment
    #         self.min_max = min_max
    #         self.mean = [0.] if mean is None else [mean]
    #         self.std = [1.] if std is None else [std]
    #         self.max_pix_val = 1.0 if min_max else 255.
    #
    #     @staticmethod
    #     def min_max_norm(img):
    #         normalize = (img - img.min()) / (img.max() - img.min())
    #         return normalize.astype("float32")
    #
    #     def transforms(self, image, mask):
    #         # Making sure mask is binary
    #         mask = (mask / mask.max()).astype("float32")
    #         normalize = tt.Normalize(self.mean, self.std)
    #         to_tensor = tt.ToTensor()
    #
    #         if self.min_max:
    #             # Divide images with their min and max pixels values
    #             image = self.min_max_norm(image)
    #             image = from_numpy(image)
    #             mask = tf.to_tensor(mask)
    #             # image = normalize(image)
    #         else:
    #             image = to_tensor(image)
    #             mask = tf.to_tensor(mask)
    #             # image = normalize(image)
    #
    #         if self.augment:
    #             # Random horizontal flipping
    #             if random.random() > 0.5:
    #                 image = tf.hflip(image)
    #                 mask = tf.hflip(mask)
    #             # Random vertical flipping
    #             if random.random() > 0.5:
    #                 image = tf.vflip(image)
    #                 mask = tf.vflip(mask)
    #             # # Random Rotate
    #             # if random.random() > 0.5:
    #             #     image = tf.rotate(image, angle=2.0)
    #             #     mask = tf.rotate(mask, angle=2.0)
    #
    #             if random.random() > 0.5:
    #                 brightness_factor = random.uniform(0.4, 1.4)
    #                 image = tf.adjust_brightness(image, brightness_factor)
    #
    #         image = normalize(image)
    #
    #         image = F.pad(image, (3, 3, 0, 0), value=0)
    #         mask = F.pad(mask, (3, 3, 0, 0), value=0)
    #         return image, mask
    #
    #     def __len__(self):
    #         return len(self.images)
    #
    #     def __getitem__(self, index):
    #         image = self.images[index]
    #         mask = self.masks[index]
    #
    #         image, mask = self.transforms(image, mask)
    #         return image, mask
    #
    #         # if self.augment:
    #         #     compose_img = tt.Compose([
    #         #         tt.ToTensor(),
    #         #         tt.Normalize(self.mean, self.std),
    #         #         # tt.RandomBrightness(limit=(-1, 1), p=1),
    #         #         tt.RandomHorizontalFlip(p=0.5),
    #         #         tt.RandomVerticalFlip(p=0.5),
    #         #         tt.RandomRotation(degrees=2),
    #         #     ])
    #         # else:
    #         #     compose_img = tt.Compose([
    #         #         tt.ToTensor(),
    #         #         # tt.Normalize(self.mean, self.std)
    #         #     ])
    #         #
    #         # image = compose_img(img)
    #         # mask = compose_img(msk)
    #
    #         # transformed = compose_obj(image=img, mask=msk)
    #         # img_tensor = from_numpy(transformed["image"])
    #         # mask_tensor = from_numpy(transformed["mask"])
    #         # # Add an extra channel [H, W] --> [1, H, W]
    #         # img_tensor = img_tensor.unsqueeze(0)
    #         # Apply padding to the images and masks (250 --> 256)
    #         # padded_img = F.pad(img_tensor, (3, 3, 0, 0), value=0)
    #         # padded_msk = F.pad(mask_tensor, (3, 3, 0, 0), value=0)

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
            img_tensor = from_numpy(transformed["image"])
            # Add an extra channel [H, W] --> [1, H, W]
            img_tensor = img_tensor.unsqueeze(0)
            # Apply padding to the images and masks (250 --> 256)
            padded_img = F.pad(img_tensor, (3, 3, 0, 0), value=0)
            return padded_img
