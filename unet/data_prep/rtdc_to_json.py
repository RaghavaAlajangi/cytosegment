from pathlib import Path
from PIL import Image
import random

import click
import dclab
import numpy as np

from .inferencer import load_model
from .inferencer import get_bnet_predictions, get_unet_prediction
from .inferencer import get_transformed_image
from .inferencer import extract_event_masks, extract_patch_tensors
from unet.data_prep.labelMap import id_to_class
from .labelme_utils import create_json

models_path = Path(__file__).parents[1] / "models"


def save_image(image, path):
    img_pil = Image.fromarray(image)
    img_pil.save(str(path))


def get_image_cor(image, image_bg):
    return np.array(image, dtype=int) - image_bg + image_bg.mean()


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
              help="Path to save output files (images, JSON files). "
                   "If it is not given, script creates new folder "
                   "based on `path_in`")
@click.option("--bb_ckp_path",
              type=click.Path(exists=True,
                              dir_okay=False,
                              resolve_path=True,
                              path_type=Path),
              help="Path to bloody_bunny model checkpoint. If it is not "
                   "given default checkpoint is taken from model folder")
@click.option("--unet_ckp_path",
              type=click.Path(exists=True,
                              dir_okay=False,
                              resolve_path=True,
                              path_type=Path),
              help="Path to unet model checkpoint. If it is not given "
                   "default checkpoint is taken from model folder")
@click.option("--min_score", "-s", type=float, default=0.5,
              help="Specify minimum probability of `ml_score` feature")
@click.option("--ml_feat_kv", "-m",
              multiple=True,
              help="KEY=VALUE argument for cell_type and num_samples pair "
                   "that needs to be extracted from RTDC dataset. "
                   "i.e `ml_score_r1f=10` If it is not given, `ml_score` "
                   "features in dataset are used to generate labelme samples "
                   "(by default 50 samples from each type)")
@click.option("--use_cuda", "-c", is_flag=True,
              help="Specify whether cuda device available or not")
def extract(path_in, path_out, min_score=0.5, ml_feat_kv=None,
            bb_ckp_path=None, unet_ckp_path=None, use_cuda=False):
    # Get the default model checkpoints
    if bb_ckp_path is None:
        bb_ckp_path = models_path / "new_mnet_minmax.pth"

    if unet_ckp_path is None:
        unet_ckp_path = models_path / "E38_validAcc_9838_jit.ckp"

    # Get the dataset name (i.e ../{dataset_name}/M001_data.rtdc)
    ds_name = path_in.parts[-2]
    if path_out is None:
        path_out = path_in.with_name(f"{ds_name}_labelme")
        path_out.mkdir(parents=True, exist_ok=True)
    else:
        path_out = path_out / f"{ds_name}_labelme"
        path_out.mkdir(parents=True, exist_ok=True)

    # Load the models
    bb = load_model(str(bb_ckp_path), use_cuda)
    unet = load_model(str(unet_ckp_path), use_cuda)

    # Read the RTDC dataset
    rtdc_ds = dclab.new_dataset(path_in)

    # If ml_score features are not specified in CLI, labelme files are
    # created based on available bloody_bunny or mnet predictions in the
    # dataset (by default 50 samples from each ml_score type)
    if ml_feat_kv is None:
        feats = [f for f in rtdc_ds.features if "ml_score_" in f]
        ml_feat_kv = {k: 50 for k in feats}
    else:
        ml_feat_kv = {k: int(v) for k, v in [n.split("=") for n in ml_feat_kv]}

    for ml_feat, num_samples in ml_feat_kv.items():
        if ml_feat in rtdc_ds.features:
            # Create directories to save labelme, image, and image_bg files
            labelme_dir = path_out / ml_feat / "labelme"
            image_dir = path_out / ml_feat / "image"
            image_bg_dir = path_out / ml_feat / "image_bg"
            unet_pred_dir = path_out / ml_feat / "unet_predicts"

            labelme_dir.mkdir(parents=True, exist_ok=True)
            image_dir.mkdir(parents=True, exist_ok=True)
            image_bg_dir.mkdir(parents=True, exist_ok=True)
            unet_pred_dir.mkdir(parents=True, exist_ok=True)

            ml_score = np.array(rtdc_ds[ml_feat])
            # Get the indices of the events that have minimum probability
            # as specified in CLI or default (0.5) i.e [0.5 <= ml_score <= max]
            feat_idx = np.where((ml_score >= min_score) &
                                (ml_score <= ml_score.max()))[0]

            if len(feat_idx) != 0:
                counter = 0
                for idx in random.sample(range(0, len(feat_idx)),
                                         len(feat_idx) - 1):
                    img = rtdc_ds["image"][idx]
                    img_bg = rtdc_ds["image_bg"][idx]
                    frm = rtdc_ds["frame"][idx]
                    ido = rtdc_ds["index_online"][idx]

                    # Get transformed image tensor
                    img_tensor = get_transformed_image(img, mean=0.6735,
                                                       std=0.1342)

                    # Get the unet prediction of the image
                    unet_pred = get_unet_prediction(img_tensor, unet, use_cuda)

                    # Split unet prediction into individual event masks
                    event_masks = extract_event_masks(unet_pred)

                    # Compute img_cor feature
                    img_cor = get_image_cor(img, img_bg)

                    # Get the transformed image_cor tensor
                    img_cor_tensor = get_transformed_image(img_cor, mean=0.5,
                                                           std=0.25)

                    # Extract cell patches based on image_corr and event masks
                    patch_tens = extract_patch_tensors(img_cor_tensor,
                                                       event_masks)

                    # Predict cell patches with bloody bunny
                    cell_preds = get_bnet_predictions(patch_tens, bb, use_cuda)

                    # Label mapping
                    cell_labels = [id_to_class[i] for i in cell_preds]

                    # Create base path with dataset name, frameNum,
                    # and indexNum to save images, image_bgs, json files
                    base_path = f"{ds_name}_frm_{int(frm)}_idx_{int(ido)}"

                    # Create json_path to save labelme files
                    json_path = labelme_dir / base_path

                    # Create img_path to save image
                    img_path = image_dir / (base_path + "_img.png")

                    # Create img_bg_path to save image_bg
                    img_bg_path = image_bg_dir / (base_path + "_img_bg.png")

                    # Create unet_predict_path to save unet predictions
                    unet_pred_path = unet_pred_dir / (base_path + "_pred.bmp")

                    # Save image in image directory
                    save_image(img, img_path)

                    # Save image_bg in image_bg directory
                    save_image(img_bg, img_bg_path)

                    # Save unet_predict in unet_pred_dir directory
                    save_image(unet_pred, unet_pred_path)

                    # Create json file with cell contours and labels
                    create_json(img, unet_pred, cell_labels, json_path)

                    counter += 1
                    print(f"{ml_feat} --> {counter} files", end="\r",
                          flush=True)
                    if counter == num_samples:
                        print(f"{ml_feat} --> {counter} files")
                        break
            else:
                print(f"Zero samples in the probability range "
                      f"[{min_score}, {ml_score.max()}]")
        else:
            print(f"Dataset does not contains {ml_feat} feature!")


if __name__ == "__main__":
    extract()
