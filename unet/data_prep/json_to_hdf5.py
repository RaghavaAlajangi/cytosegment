from pathlib import Path
from PIL import Image

import click
import h5py
import numpy as np

from .labelme_utils import json_to_mask


def create_hdf5(images, image_bgs, masks, filename="segment_dataset.hdf5"):
    compression = "gzip"
    with h5py.File(filename, "a") as h:
        imset = h.create_dataset(name="events/image",
                                 data=np.array(images),
                                 shape=np.array(images).shape,
                                 dtype=np.array(images).dtype,
                                 compression=compression
                                 )
        imset.attrs.create("CLASS", "IMAGE", dtype="S6")
        imset.attrs.create("IMAGE_SUBCLASS", "IMAGE_GRAYSCALE", dtype="S16")
        imset.attrs.create("IMAGE_VERSION", "1.2", dtype="S4")
        imset.attrs.create("INTERLACE_MODE", "INTERLACE_PIXEL", dtype="S16")
        imbgset = h.create_dataset(name="events/image_bg",
                                   data=np.array(image_bgs),
                                   shape=np.array(image_bgs).shape,
                                   dtype=np.array(image_bgs).dtype,
                                   compression=compression
                                   )
        imbgset.attrs.create("CLASS", "IMAGE", dtype="S6")
        imbgset.attrs.create("IMAGE_SUBCLASS", "IMAGE_GRAYSCALE", dtype="S16")
        imbgset.attrs.create("IMAGE_VERSION", "1.2", dtype="S4")
        imbgset.attrs.create("INTERLACE_MODE", "INTERLACE_PIXEL", dtype="S16")
        mset = h.create_dataset(name="events/mask",
                                data=np.array(masks),
                                shape=np.array(masks).shape,
                                dtype=np.array(masks).dtype,
                                compression=compression
                                )
        mset.attrs.create("CLASS", "IMAGE", dtype="S6")
        mset.attrs.create("IMAGE_SUBCLASS", "IMAGE_GRAYSCALE", dtype="S16")
        mset.attrs.create("IMAGE_VERSION", "1.2", dtype="S4")
        mset.attrs.create("INTERLACE_MODE", "INTERLACE_PIXEL", dtype="S16")


def get_sorted_files(path_in):
    valid_json_files = []
    valid_img_files = []
    valid_img_bg_files = []

    # Get the interpolated json files from the path_in directory
    json_files = [p for p in Path(path_in).rglob("*interpolated.json") if
                  p.is_file()]
    sorted_json_files = sorted(json_files)

    # Get the image files from the path_in directory
    img_files = [p for p in Path(path_in).rglob("*img.png") if p.is_file()]
    sorted_img_files = sorted(img_files)

    # Get the image_bg files from the path_in directory
    img_bg_files = [p for p in Path(path_in).rglob("*img_bg.png") if
                    p.is_file()]
    sorted_img_bg_files = sorted(img_bg_files)

    # Create a list of image stem names
    img_stems = [str(f.name).split('img.png')[0] for f in sorted_img_files]

    # Create a list of image_bg stem names
    img_bg_stems = [str(f.name).split('img_bg.png')[0] for f in
                    sorted_img_bg_files]

    for json_file in sorted_json_files:
        json_stem = str(json_file.name).split('img_interpolated')[0]
        if json_stem in img_stems and json_stem in img_bg_stems:
            # Get the indices for similar file names as json file
            img_idx = img_stems.index(json_stem)
            img_bg_idx = img_bg_stems.index(json_stem)
            # Get the image and image_bg files based in indices
            img_file = sorted_img_files[img_idx]
            img_bg_file = sorted_img_bg_files[img_bg_idx]
            # Get the valid file names of all the required types
            valid_json_files.append(json_file)
            valid_img_files.append(img_file)
            valid_img_bg_files.append(img_bg_file)
    return valid_json_files, valid_img_files, valid_img_bg_files


@click.command(help="This script helps to create HDF5 file from "
                    "JSON files.")
@click.option("--path_in",
              type=click.Path(exists=True,
                              dir_okay=True,
                              resolve_path=True,
                              path_type=Path),
              help="Path to 'img.png', 'img_bg.png', and 'img.json' files "
                   "directory. Make sure, the path you are providing has all "
                   "type of files. (script collects files recursively)")
@click.option("--path_out",
              type=click.Path(dir_okay=False,
                              writable=True,
                              resolve_path=True,
                              path_type=Path),
              help="Optional! Path to save output (.hdf5) file")
def main(path_in, path_out=None):
    if path_out is None:
        path_out = path_in.with_name("segm_dataset.hdf5")

    file_lists = get_sorted_files(path_in)

    files_lengths = list(map(len, file_lists))
    if any(files_lengths) != 0:
        if all(x == files_lengths[0] for x in files_lengths):
            # Create images and masks out of json files
            masks = json_to_mask(file_lists[0])
            images = [np.array(Image.open(f)) for f in file_lists[1]]
            image_bgs = [np.array(Image.open(f)) for f in file_lists[2]]
            # Create final hdf5 file
            create_hdf5(images, image_bgs, masks, filename=path_out)
            print("HDF5 file is created!")
        else:
            print("Given path has missing files!")
    else:
        print("Could not find the required files!")


if __name__ == "__main__":
    main()
