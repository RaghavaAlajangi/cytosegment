from pathlib import Path

import click
import h5py
import numpy as np

from labelme_utils import json_to_mask


def create_hdf5(images, masks, filename="segment_dataset.hdf5"):
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


@click.command(help="This script helps to create HDF5 file from "
                    "Labelme JSON files")
@click.argument("path_in",
                type=click.Path(exists=True,
                                dir_okay=True,
                                resolve_path=True,
                                path_type=Path))
@click.argument("path_out",
                required=False,
                type=click.Path(dir_okay=False,
                                writable=True,
                                resolve_path=True,
                                path_type=Path))
def main(path_in, path_out=None):
    if path_out is None:
        path_out = path_in.with_name("segm_dataset.hdf5")

    # Get the JSON files in the path_in folder
    json_files = [p for p in Path(path_in).rglob("*.json") if p.is_file()]
    # Create images and masks out of json files
    images, masks = json_to_mask(json_files)
    # Create final hdf5 file
    create_hdf5(images, masks, filename=path_out)
    print("HDF5 file is created!")


if __name__ == "__main__":
    main()
