import time

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from skimage.measure import find_contours
import torch
import yaml

from .ml_metrics import IoUCoeff, DiceCoeff


def load_model(ckp_path_jit, use_cuda):
    device = torch.device("cuda" if use_cuda else "cpu")
    model_jit = torch.jit.load(ckp_path_jit, map_location=device)
    model_jit.eval()
    model_jit = torch.jit.optimize_for_inference(model_jit)
    return model_jit


def inference(test_dataloader, model_path, results_path, use_cuda=True,
              save_results=False):
    params_path = yaml.safe_load(open(results_path / "train_params.yaml"))
    data_path = Path(params_path["dataset"]["data_path"]).with_suffix("")
    test_data = data_path / "testing" / "images"

    im_file_names = sorted([p.stem for p in Path(test_data).rglob("*.png")])
    im_file_names.append("Average Scores")

    if use_cuda:
        out_path = results_path / "test_results_gpu"
    else:
        out_path = results_path / "test_results_cpu"

    ioumetric = IoUCoeff()
    dicemetric = DiceCoeff(sample_wise=True)

    unet = load_model(model_path, use_cuda=use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    image_list = []
    predict_list = []
    target_list = []
    tik = time.time()

    for img_batch, lbl_batch in test_dataloader:
        img_batch = img_batch.view(-1, *img_batch.shape[2:])
        lbl_batch = lbl_batch.view(-1, *lbl_batch.shape[2:])
        img_batch = img_batch.to(device, dtype=torch.float32)
        lbl_batch = lbl_batch.to(device, dtype=torch.float32)

        predicts = unet(img_batch)

        predict_list.append(predicts)
        target_list.append(lbl_batch)
        image_list.append(img_batch)

    inference_time = time.time() - tik
    inf_time_per_img = inference_time / len(test_dataloader.dataset)
    if use_cuda:
        print("Inference time (gpu)/image:", inf_time_per_img)
    else:
        print("Inference time (cpu)/image:", inf_time_per_img)

    image_torch = torch.cat(image_list, dim=0)
    predict_torch = torch.cat(predict_list, dim=0)
    target_torch = torch.cat(target_list, dim=0)

    iou_scores = ioumetric(predict_torch, target_torch)
    dice_scores = dicemetric(predict_torch, target_torch)
    iou_scores = iou_scores.detach().cpu().numpy()
    dice_scores = dice_scores.detach().cpu().numpy()

    iou_scores = np.append(iou_scores, [np.mean(iou_scores)])
    dice_scores = np.append(dice_scores, [np.mean(dice_scores)])

    if save_results:
        out_path.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(10, 5))
        plt.hist(dice_scores, bins=50, label="Dice Scores")
        plt.legend()
        fig.tight_layout()
        fig.savefig(out_path / "dice_scores_dis.png")
        plt.close()

        image_numpy = image_torch.squeeze(1).detach().cpu().numpy()
        mask_numpy = target_torch.squeeze(1).detach().cpu().numpy()
        predict_numpy = torch.sigmoid(predict_torch).squeeze(
            1).detach().cpu().numpy()

        for n in range(len(image_numpy)):
            img = image_numpy[n]
            msk = mask_numpy[n]
            pred = predict_numpy[n]
            iou = iou_scores[n] * 100
            dice = dice_scores[n] * 100
            im_name = im_file_names[n]

            pred_cnt = find_contours(pred, 0.8)
            msk_cnt = find_contours(msk, 0.8)

            fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))

            axs[0, 0].imshow(img, "gray")
            axs[0, 0].axis("off")
            axs[0, 1].imshow(msk, "gray")
            axs[0, 1].set_title("Ground truth")
            axs[0, 1].axis("off")

            axs[0, 2].imshow(pred, "gray")
            axs[0, 2].set_title(f"Pred (IoU: {iou:.3f}, Dice: {dice:.3f})")
            axs[0, 2].axis("off")
            gs = axs[1, 1].get_gridspec()

            # remove the underlying axes
            for ax in axs[1, :]:
                ax.remove()
            axbig = fig.add_subplot(gs[1, :])
            axbig.imshow(img, "gray")
            axbig.axis("off")
            for pc in pred_cnt:
                axbig.plot(pc[:, 1], pc[:, 0], "r", linewidth=2)
            for mc in msk_cnt:
                axbig.plot(mc[:, 1], mc[:, 0], "g", linewidth=2)

            red_patch = mpatches.Patch(color="red", label="Prediction")
            green_patch = mpatches.Patch(color="green", label="Ground Truth")
            plt.legend(handles=[red_patch, green_patch])

            fig.tight_layout()
            fig.savefig(
                out_path / f"{im_name}_iou_{iou:.1f}_dice_{dice:.1f}.png")
            plt.close()
    return inf_time_per_img, iou_scores, dice_scores, im_file_names
