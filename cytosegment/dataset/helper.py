from pathlib import Path
from PIL import Image
import random
import zipfile


def verify_image_file(file_path):
    """Verify if the image is valid."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        print(f"Excluding corrupted file: ({file_path})")
        return False


def intersection_of_images_and_masks(image_paths, mask_paths):
    # Convert paths to sets of filenames for intersection
    # .stem gets the filename without extension
    image_filenames = {img.stem for img in image_paths}
    mask_filenames = {mask.stem for mask in mask_paths}

    # Find the common files (intersection)
    common_filenames = image_filenames & mask_filenames

    # Filter the images and masks based on the intersection
    filtered_images = [img for img in image_paths if
                       img.stem in common_filenames]
    filtered_masks = [mask for mask in mask_paths if
                      mask.stem in common_filenames]

    return filtered_images, filtered_masks


def read_data_files(data_path, seed=42, shuffle=False):
    """Reads image and mask data from the specified path."""
    image_dir = Path(data_path) / "images"
    mask_dir = Path(data_path) / "masks"

    image_paths = sorted([p for p in image_dir.rglob("*.png") if p.is_file()])
    mask_paths = sorted([p for p in mask_dir.rglob("*.png") if p.is_file()])

    # Verify and filter corrupted files
    valid_img_list = [img for img in image_paths if verify_image_file(img)]
    valid_msk_list = [msk for msk in mask_paths if verify_image_file(msk)]

    if len(valid_img_list) != len(valid_msk_list):
        print(f"Warning: After verification, the number of valid images "
              f"({len(valid_img_list)}) and masks ({len(valid_msk_list)}) is "
              f"different.")
        valid_img_list, valid_msk_list = intersection_of_images_and_masks(
            valid_img_list, valid_msk_list)
        print(f"Using {len(valid_img_list)} common valid images and masks.")

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
