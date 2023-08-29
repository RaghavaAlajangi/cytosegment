import time

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from skimage.measure import find_contours
import torch
import yaml

from .cli.cli_inference import load_model
from .ml_metrics import IoUCoeff, DiceCoeff
from .ml_dataset import get_data_files, read_data, create_dataloaders, \
    crop_pad_data, UNetDataset


def inference(model_path, results_path, dataset, use_cuda=True,
              bsize=8, save_results=False):
    mean = dataset.mean[0]
    std = dataset.std[0]
    min_max = dataset.min_max
    img_size = dataset.images[0].shape

    params_path = yaml.safe_load(open(results_path / "train_params.yaml"))
    data_path = Path(params_path["dataset"]["data_path"]).with_suffix('')
    test_data = data_path / "testing"

    if use_cuda:
        out_path = results_path / "test_results_gpu"
    else:
        out_path = results_path / "test_results_cpu"

    img_files, msk_files = get_data_files(test_data, shuffle=False)
    im_file_names = [path.stem for path in img_files]
    im_file_names.append("Average Scores")
    images, masks = read_data(img_files, msk_files)

    resized_data = [crop_pad_data(img, msk, img_size) for img, msk in
                    zip(images, masks)]

    images = [sublist[0] for sublist in resized_data]
    masks = [sublist[1] for sublist in resized_data]

    test_dataset = UNetDataset(images, masks, min_max=min_max,
                               mean=mean, std=std)

    data_dict = {"test": test_dataset}
    dataloader = create_dataloaders(data_dict,
                                    batch_size=bsize,
                                    num_workers=0)["test"]

    ioumetric = IoUCoeff()
    dicemetric = DiceCoeff(sample_wise=True)

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

    inference_time = time.time() - tik
    inf_time_per_img = inference_time / test_dataset.__len__()
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
        fig = plt.figure(figsize=(8, 4))
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
            iou = iou_scores[n]
            dice = dice_scores[n]
            im_name = im_file_names[n]

            pred_cnt = find_contours(pred, 0.8)
            msk_cnt = find_contours(msk, 0.8)

            fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))

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
            fig.savefig(
                out_path / f"{im_name}_iou_{iou * 100:.1f}_dice_{dice * 100:.1f}.png")
            plt.close()

    return test_dataset.__len__(), inf_time_per_img, iou_scores, dice_scores, im_file_names
