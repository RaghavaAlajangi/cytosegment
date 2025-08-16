import csv
import time
from ast import literal_eval as asteval

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.measure import find_contours

from ..training.metrics import DiceCoeff, IoUCoeff


def load_model(jit_model_path, device):
    """Load a PyTorch model from a JIT checkpoint file and its associated
    metadata.

    Parameters
    ----------
    jit_model_path: str or pathlib.Path
        Load the model checkpoint
    device: str or device
        Determine device where you want to run the model.
    Returns
    -------
    model_jit : torch.jit.ScriptModule
        The loaded PyTorch model stored as a TorchScript module, optimized
        for inference.

    model_meta : dict or None
        The metadata associated with the loaded model, if available.
        If no metadata is found, returns `None`.
    """
    # Define a mapping dict to load model metadata
    extra_files = {"meta": ""}
    # Load model and its metadata
    model_jit = torch.jit.load(
        jit_model_path, _extra_files=extra_files, map_location=device
    )
    model_meta = None
    # check if model checkpoint has any metadata
    if extra_files["meta"]:
        decode_meta = extra_files["meta"].decode("utf-8")
        # Convert bytes representation to a dictionary
        model_meta = asteval(decode_meta)

    model_jit.eval()
    model_jit = torch.jit.optimize_for_inference(model_jit)

    return model_jit, model_meta


class BaseInference:
    def __init__(self, jit_model_path, use_cuda):
        """
        Parameters
        ----------
        jit_model_path: str or pathlib.Path
            Load the model checkpoint
        use_cuda: bool
            Whether you want to use cuda device.
        """
        self.jit_model_path = jit_model_path
        self._use_cuda = use_cuda
        self.device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )
        self.model, _ = load_model(jit_model_path, self.device)

        # Define metrics
        self.iou_metric = IoUCoeff()
        self.dice_metric = DiceCoeff(sample_wise=True)

    @property
    def use_cuda(self):
        return self._use_cuda

    @use_cuda.setter
    def use_cuda(self, value):
        """When use_cuda changes device and model attributes get updated."""
        self._use_cuda = value
        self.device = torch.device(
            "cuda" if self._use_cuda and torch.cuda.is_available() else "cpu"
        )
        self.model, _ = load_model(self.jit_model_path, self.device)

    def run_inference_loop(self, dataloader):
        """Run inferencing on batches and return predictions."""
        image_torch_list, predict_torch_list, mask_torch_list = [], [], []

        tik = time.time()
        with torch.no_grad():  # Disable gradients for inference
            for image_batch, mask_batch in dataloader:
                # Move only image batch to device
                image_batch = image_batch.to(self.device, dtype=torch.float32)

                predict_batch = self.model(image_batch)
                # Bring image and prediction batches back to cpu device
                image_torch_list.append(image_batch.detach().cpu())
                predict_torch_list.append(predict_batch.detach().cpu())
                mask_torch_list.append(mask_batch)

                # Delete tensors and clear cache after each batch
                del image_batch, predict_batch
                torch.cuda.empty_cache()

        inference_time = time.time() - tik

        inference_per_image = inference_time / len(dataloader.dataset)

        image_file_names = [p.stem for p in dataloader.dataset.image_paths]
        image_file_names.append("Average Scores")

        image_torch = torch.cat(image_torch_list, dim=0)
        predict_torch = torch.cat(predict_torch_list, dim=0)
        mask_torch = torch.cat(mask_torch_list, dim=0)

        iou_scores = self.iou_metric(predict_torch, mask_torch).numpy()
        dice_scores = self.dice_metric(predict_torch, mask_torch).numpy()

        # Compute and append mean values of the scores
        iou_scores = np.append(iou_scores, [np.mean(iou_scores)])
        dice_scores = np.append(dice_scores, [np.mean(dice_scores)])

        return {
            # Return processed images, masks, and predictions
            "images": image_torch.squeeze(1).detach().cpu().numpy(),
            "predicts": predict_torch.squeeze(1).detach().cpu().numpy(),
            "masks": mask_torch.squeeze(1).detach().cpu().numpy(),
            # Return scores
            "inference_time": inference_per_image,
            "iou": iou_scores,
            "dice": dice_scores,
            # Return test image file names (useful for plotting)
            "image_names": image_file_names,
        }

    @staticmethod
    def save_metrics(inference_output, out_path):
        """Dump inference scores to a CSV."""
        scores = {
            "image_names": inference_output["image_names"],
            "iou_scores": inference_output["iou"],
            "dice_scores": inference_output["dice"],
        }
        with open(out_path / "test_scores.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(scores.keys())
            writer.writerows(zip(*scores.values()))

    @staticmethod
    def plot_predictions(inference_output, out_path):
        """Generate and save the prediction visualization."""
        images = inference_output["images"]
        masks = inference_output["masks"]
        predicts = inference_output["predicts"]
        iou_scores = inference_output["iou"]
        dice_scores = inference_output["dice"]
        image_names = inference_output["image_names"]
        for n in range(len(images)):
            img, msk, pred = images[n], masks[n], predicts[n]
            iou, dice = iou_scores[n] * 100, dice_scores[n] * 100
            name = image_names[n]
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
            fig.savefig(out_path / f"{name}_iou_{iou:.1f}_dice_{dice:.1f}.png")
            plt.close()
