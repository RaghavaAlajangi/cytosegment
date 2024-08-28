from pathlib import Path
import random
import zipfile


def read_data_files(data_path, seed=42, shuffle=False):
    """Reads image and mask data from the specified path."""
    image_dir = Path(data_path) / "images"
    mask_dir = Path(data_path) / "masks"

    image_paths = sorted([p for p in image_dir.rglob("*.png") if p.is_file()])
    mask_paths = sorted([p for p in mask_dir.rglob("*.png") if p.is_file()])

    assert len(image_paths) == len(mask_paths)

    if shuffle:
        random.seed(seed)
        image_paths, mask_paths = zip(
            *random.sample(list(zip(image_paths, mask_paths)),
                           len(image_paths)))

    return image_paths, mask_paths


def split_data(images, masks, valid_size=0.2):
    """Splits the given images and masks into training and validation sets."""
    assert len(images) == len(masks)
    train_size = int(len(images) * (1 - valid_size))
    return (images[:train_size], images[train_size:],
            masks[:train_size], masks[train_size:])


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
