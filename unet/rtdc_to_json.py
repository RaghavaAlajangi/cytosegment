from pathlib import Path
import random

import albumentations as A
import click
import cv2
import dclab
import numpy as np

from skimage import measure
import torch
from torch import from_numpy
from torchvision import transforms
from torchvision.transforms import Normalize

from labelMap import id_to_class
from labelme_utils import create_json


class ToTensorMinMax(object):
    def __init__(self):
        pass

    def __call__(self, image):
        min_image, max_image = image.min(), image.max()
        tensor = torch.tensor((image - min_image)
                              / (max_image - min_image),
                              dtype=torch.float32)
        return tensor.unsqueeze(0)


def load_model(ckp_path_jit, use_cuda=True):
    device = torch.device("cuda" if use_cuda else "cpu")
    model_jit = torch.jit.load(ckp_path_jit, map_location=device)
    model_jit.eval()
    model_jit = torch.jit.optimize_for_inference(model_jit)
    return model_jit


def crop_image(img, center_cords, crop_size=80,
               max_x=250, max_y=80):
    sh = crop_size // 2
    px, py = center_cords
    px = np.minimum(np.maximum(px, sh), max_x - sh).astype(np.int16)
    py = np.minimum(np.maximum(py, sh), max_y - sh).astype(np.int16)
    img_cropped = img[py - sh:py + sh, px - sh:px + sh]
    return img_cropped


def extract_patch_tensors(image_corr, masks):
    transform = transforms.Compose([ToTensorMinMax(),
                                    Normalize(mean=(0.5), std=(0.25))
                                    ])
    patch_tensors = []
    for m in masks:
        M = measure.moments(m)
        px = M[0, 1] / M[0, 0]
        py = M[1, 0] / M[0, 0]
        # Apply transform --> get the org image tensor
        trans_img_corr = transform(image_corr)
        trans_img_corr = trans_img_corr.squeeze(0)
        # Crop the org img tensor into patches based on coords
        patch_tensor = crop_image(trans_img_corr, (px, py))
        # Add an extra channel [H, W] --> [1, H, W]
        patch_tensor = patch_tensor.unsqueeze(0)
        patch_tensors.append(patch_tensor)
    return patch_tensors


def extract_masks_polygons(unet_pred):
    polygons = measure.find_contours(unet_pred, 0.2,
                                     fully_connected="low",
                                     positive_orientation="high")

    pred_bin = (unet_pred > 0.4).astype("int8")
    segm, num = measure.label(pred_bin, background=0, return_num=True)
    masks = []
    for n in range(num):
        mask_n = segm == n + 1
        masks.append(mask_n)
    return masks, polygons


def min_max_norm(img):
    return (img - img.min()) / (img.max() - img.min())


def get_unet_prediction(image, model, use_cuda=True,
                        mean=None, std=None):
    std = [0.1342] if std is None else std
    mean = [0.6735] if mean is None else std

    device = torch.device("cuda" if use_cuda else "cpu")
    image = min_max_norm(image)
    compose_obj = A.Compose([A.Normalize(mean, std, max_pixel_value=1.)])
    transformed = compose_obj(image=image)
    img_tensor = from_numpy(transformed["image"])
    im_ten = img_tensor.unsqueeze(0).unsqueeze(0)
    im_ten = im_ten.to(device)
    pred = model(im_ten)
    pred = pred.squeeze(0).squeeze(0).detach().cpu().numpy()
    return pred


def get_bnet_predictions(patches, mnet, use_cuda=True):
    device = torch.device("cuda" if use_cuda else "cpu")
    # Stack list of patches into batch
    # [(1, 80, 80), (1, 80, 80)] --> [2, 1, 80, 80]
    min_patch_batch = torch.stack(patches, dim=0)
    input_tensor = min_patch_batch.to(device)
    output = mnet(input_tensor)
    # Get the probabilities of the classes
    output = torch.softmax(output, dim=1)
    # Get the maximum probability index
    labels = torch.argmax(output, dim=-1)
    # Convert torch labels into int
    preds = [int(i) for i in labels]
    return preds


@click.command(help="This script helps to create JSON files from RTDC dataset")
@click.option("--path_in",
              type=click.Path(exists=True,
                              dir_okay=False,
                              resolve_path=True,
                              path_type=Path),
              help="Path to RTDC dataset (.rtdc)")
@click.option("--path_out",
              type=click.Path(dir_okay=True,
                              writable=True,
                              resolve_path=True,
                              path_type=Path),
              help="Path to save output files (images, JSON files)")
@click.option("--bb_ckp_path",
              type=click.Path(exists=True,
                              dir_okay=False,
                              resolve_path=True,
                              path_type=Path),
              help="Path to bloody_bunny model checkpoint")
@click.option("--unet_ckp_path",
              type=click.Path(exists=True,
                              dir_okay=False,
                              resolve_path=True,
                              path_type=Path),
              help="Path to unet model checkpoint")
@click.option("--kwrags", "-k",
              multiple=True,
              required=True,
              help="KEY=VALUE argument for cell_type and num_samples "
                   "that needs to be extracted from RTDC dataset")
@click.option("--is_cuda", "-c", is_flag=True,
              help="Specify whether cuda device available or not")
def main(path_in, path_out, bb_ckp_path,
         unet_ckp_path, kwrags, is_cuda=False):
    # Get the key and value arguments
    ml_score_feats = {}
    for key, value in [a.split("=") for a in kwrags]:
        ml_score_feats[key] = int(value)
    assert len(ml_score_feats) != 0

    # Get the dataset name (i.e ../{dataset_name}/M001_data.rtdc)
    DS_NAME = path_in.parts[-2]
    if path_out is None:
        path_out = path_in.with_name(f"{DS_NAME}_labelme")
        path_out.mkdir(parents=True, exist_ok=True)
    else:
        path_out = path_out / f"{DS_NAME}_labelme"
        path_out.mkdir(parents=True, exist_ok=True)

    # Load the models
    bb = load_model(str(bb_ckp_path), use_cuda=is_cuda)
    unet = load_model(str(unet_ckp_path), use_cuda=is_cuda)

    rtdc_ds = dclab.new_dataset(path_in)
    for ml_feat, num_cells in ml_score_feats.items():
        if ml_feat in rtdc_ds.features:
            # Create folders to save labelme and image_bg files
            path_labelme = path_out / ml_feat / "labelme"
            path_image_bg = path_out / ml_feat / "image_bg"

            path_labelme.mkdir(parents=True, exist_ok=True)
            path_image_bg.mkdir(parents=True, exist_ok=True)

            ml_score = np.array(rtdc_ds[ml_feat])
            feat_idx = np.where(((ml_score > 0.1) & (ml_score < 0.9)))[0]

            counter = 0
            for idx in random.sample(range(0, len(feat_idx)),
                                     len(feat_idx) - 1):
                img = rtdc_ds["image"][idx]
                img_bg = rtdc_ds["image_bg"][idx]
                frm = rtdc_ds["frame"][idx]
                ido = rtdc_ds["index_online"][idx]

                # Create img_corr feature
                img_cor = np.array(img, dtype=int) - img_bg + img_bg.mean()

                # Get the unet prediction of image
                segm_pred = get_unet_prediction(img, unet)

                # Split unet prediction into individual cell masks
                msks, polygons = extract_masks_polygons(segm_pred)

                # Extract cell patches based on image, image_bg, and mask
                patch_tens = extract_patch_tensors(img_cor, msks)

                # Predict cell patches with bloody bunny
                cell_preds = get_bnet_predictions(patch_tens, bb)

                # Label mapping
                cell_labels = [id_to_class[i] for i in cell_preds]

                # Create img_path with dataset name, frameNum, and indexNum
                img_path = path_labelme / f"{DS_NAME}_frm_{int(frm)}" \
                                          f"_idx_{int(ido)}_img.png"

                # Create img_bg_path with dataset name, frameNum, and indexNum
                img_bg_path = path_image_bg / f"{DS_NAME}_frm_{int(frm)}" \
                                              f"_idx_{int(ido)}_img_bg.png"

                # Save image_bg
                cv2.imwrite(str(img_bg_path), img_bg)

                # Create json file with cell contours and labels
                create_json(img, polygons, cell_labels, img_path)

                counter += 1
                print(f"{ml_feat} --> {counter} files", end="\r", flush=True)
                if counter == num_cells:
                    print(f"{ml_feat} --> {counter} files")
                    break


if __name__ == "__main__":
    main()
