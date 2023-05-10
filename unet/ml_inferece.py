import time

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.measure import find_contours
import torch

from .cli.cli_inference import load_model
from .ml_metrics import IoUCoeff, DiceCoeff
from .ml_dataset import *

data_path = "data/training_testing_set_w_beads/testing"

use_cuda = True
mean = [0.486]
std = [0.115]
bsize = 16
min_max = False


def inference(model_path, results_path):
    out_path = results_path / "test_results"
    out_path.mkdir(parents=True, exist_ok=True)

    images, masks = read_suffle_data(data_path)
    images, masks = crop_data(images, masks)

    test_dataset = UNetDataset(images, masks, min_max=min_max,
                               mean=mean, std=std)

    data_dict = {"test": test_dataset}
    dataloader = create_dataloaders(data_dict,
                                    batch_size=bsize,
                                    num_workers=0)["test"]

    metric1 = IoUCoeff()
    metric2 = DiceCoeff()

    unet = load_model(model_path, use_cuda=use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    image_list = []
    predict_list = []
    target_list = []
    tik = time.time()

    for i, (img_batch, lbl_batch) in enumerate(dataloader):
        img_batch = img_batch.to(device, dtype=torch.float32)
        lbl_batch = lbl_batch.to(device, dtype=torch.float32)

        predicts = unet(img_batch)

        predict_list.append(predicts)
        target_list.append(lbl_batch)
        image_list.append(img_batch)

    image_torch = torch.cat(image_list, dim=0)
    predict_torch = torch.cat(predict_list, dim=0)
    target_torch = torch.cat(target_list, dim=0)

    iou_scores = metric1(predict_torch, target_torch)
    dice_scores = metric2(predict_torch, target_torch)
    iou_avg = float(iou_scores.detach().cpu().mean())
    dice_avg = float(dice_scores.detach().cpu().mean())

    inference_time = time.time() - tik
    inference_time_per_img = inference_time / test_dataset.__len__()

    image_numpy = image_torch.squeeze(1).detach().cpu().numpy()
    mask_numpy = target_torch.squeeze(1).detach().cpu().numpy()
    predict_numpy = torch.sigmoid(predict_torch).squeeze(
        1).detach().cpu().numpy()

    for n in range(len(image_numpy)):
        img = image_numpy[n]
        msk = mask_numpy[n]
        pred = predict_numpy[n]
        dice = iou_scores[n]
        iou = dice_scores[n]

        pred_cnt = find_contours(pred, 0.8)
        msk_cnt = find_contours(msk, 0.8)

        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

        axs[0, 0].imshow(img, 'gray')
        axs[0, 0].axis('off')
        axs[0, 1].imshow(msk, 'gray')
        axs[0, 1].set_title("Ground truth")
        axs[0, 1].axis('off')

        axs[0, 2].imshow(pred, 'gray')
        axs[0, 2].set_title(f"Pred (IoU: {iou:.3f}, Dice: {dice:.3f})")
        axs[0, 2].axis('off')
        gs = axs[1, 1].get_gridspec()

        # remove the underlying axes
        for ax in axs[1, :]:
            ax.remove()
        axbig = fig.add_subplot(gs[1, :])
        axbig.imshow(img, "gray")
        axbig.axis('off')
        for pc in pred_cnt:
            axbig.plot(pc[:, 1], pc[:, 0], "r", linewidth=2)
        for mc in msk_cnt:
            axbig.plot(mc[:, 1], mc[:, 0], "g", linewidth=2)

        red_patch = mpatches.Patch(color='red', label='Prediction')
        green_patch = mpatches.Patch(color='green', label='Ground Truth')
        plt.legend(handles=[red_patch, green_patch])

        fig.tight_layout()
        fig.savefig(out_path / f"pred{n + 1}.png")
        plt.close()

    return test_dataset.__len__(), inference_time_per_img, iou_avg, dice_avg
