from .base import BaseInference


class StandardInference(BaseInference):
    def __init__(self, model_path, dataloader, results_path, use_cuda=False):
        super().__init__(model_path, use_cuda)
        self.dataloader = dataloader
        self.results_path = results_path

    def run(self, save_plots=False):
        """Run standard inferencing and save results if needed."""
        infer_output = self.run_inference_loop(self.dataloader)

        # Define inference mode
        print(
            f"Testing folder: Inference time ({self.device.type})/image: "
            f"{infer_output['inference_time']:4f}"
        )

        if save_plots:
            # Define result directory
            out_path = self.results_path / f"test_results_{self.device.type}"
            out_path.mkdir(parents=True, exist_ok=True)

            self.save_metrics(infer_output, out_path)
            self.plot_predictions(infer_output, out_path)

        iou_mean = float(infer_output["iou"][-1])
        dice_mean = float(infer_output["dice"][-1])
        return infer_output["inference_time"], iou_mean, dice_mean
