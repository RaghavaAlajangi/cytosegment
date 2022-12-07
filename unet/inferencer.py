import albumentations as A
import numpy as np
from skimage import measure
from skimage.filters import threshold_otsu
import torch
from torch import from_numpy


def load_model(ckp_path_jit, use_cuda):
    device = torch.device("cuda" if use_cuda else "cpu")
    model_jit = torch.jit.load(ckp_path_jit, map_location=device)
    model_jit.eval()
    model_jit = torch.jit.optimize_for_inference(model_jit)
    return model_jit


def min_max_norm(img):
    return (img - img.min()) / (img.max() - img.min())


def get_transformed_image(image, mean, std):
    image = min_max_norm(image)
    compose_obj = A.Compose([A.Normalize(mean, std, max_pixel_value=1.)])
    transformed = compose_obj(image=image)
    img_tensor = from_numpy(transformed["image"])
    return img_tensor


def crop_image(img, center_cords, crop_size=80,
               max_x=250, max_y=80):
    sh = crop_size // 2
    px, py = center_cords
    px = np.minimum(np.maximum(px, sh), max_x - sh).astype(np.int16)
    py = np.minimum(np.maximum(py, sh), max_y - sh).astype(np.int16)
    img_cropped = img[py - sh:py + sh, px - sh:px + sh]
    return img_cropped


def extract_patch_tensors(trans_img_corr, event_masks):
    patch_tensors = []
    for mask in event_masks:
        moment = measure.moments(mask)
        px = moment[0, 1] / moment[0, 0]
        py = moment[1, 0] / moment[0, 0]
        # Crop the org img tensor into patches based on cords
        patch_tensor = crop_image(trans_img_corr, (px, py))
        # Add an extra channel [H, W] --> [1, H, W]
        patch_tensor = patch_tensor.unsqueeze(0)
        patch_tensors.append(patch_tensor)
    return patch_tensors


def extract_event_masks(unet_pred):
    thresh = threshold_otsu(unet_pred)
    bin_pred = (unet_pred > thresh)
    segm, num = measure.label(bin_pred, background=0, return_num=True)
    masks = []
    for n in range(num):
        mask_n = segm == n + 1
        masks.append(mask_n)
    return masks


def get_unet_prediction(image_tensor, model, use_cuda):
    device = torch.device("cuda" if use_cuda else "cpu")
    # Convert (80, 250) --> (1, 1, 80, 250)
    img_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    # Transfer the input to the GPU
    img_tensor = img_tensor.to(device)
    # Model inference
    pred = model(img_tensor)
    # Convert (1, 1, 80, 250) --> (80, 250)
    pred = pred.squeeze(0).squeeze(0)
    # Detach and bring output back to the CPU
    pred = pred.detach().cpu().numpy()
    return pred


def get_bnet_predictions(patches, mnet, use_cuda):
    device = torch.device("cuda" if use_cuda else "cpu")
    # Stack list of patches into batch:
    # [(1, 80, 80), (1, 80, 80)] --> [2, 1, 80, 80]
    min_patch_batch = torch.stack(patches, dim=0)
    # Transfer the input to the GPU
    input_tensor = min_patch_batch.to(device)
    # Model inference
    output = mnet(input_tensor)
    # Get the probabilities of the classes
    output = torch.softmax(output, dim=1)
    # Get the maximum probability index
    labels = torch.argmax(output, dim=-1)
    # Convert torch labels into int
    preds = [int(i) for i in labels]
    return preds
