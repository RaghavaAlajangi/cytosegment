from pathlib import Path

from torch.utils.data import DataLoader

from ..dataset import UNetDataset, read_data_files
from .base import BaseInference


class DividedGroupInference(BaseInference):
    def __init__(self, model_path, config, results_path, use_cuda=False):
        super().__init__(model_path, use_cuda)
        self.config = config
        self.results_path = results_path

    def run(self, save_plots=False):
        """Run inferencing with divided groups."""
        test_div_path = Path(self.config.data.path) / "testing_divided_groups"

        if not test_div_path.is_dir():
            return None

        for div_dir in test_div_path.iterdir():
            if not (div_dir / "images").is_dir():
                continue

            images_files, masks_files = read_data_files(div_dir, shuffle=False)

            dataset = UNetDataset(
                images_files,
                masks_files,
                self.config.data.img_size,
                mean=self.config.data.mean,
                std=self.config.data.std,
            )
            dataloader = DataLoader(dataset, batch_size=8, pin_memory=True)

            infer_output = self.run_inference_loop(dataloader)

            # Define inference mode
            print(
                f"{div_dir.parts[-1]} folder: "
                f"Inference time ({self.device.type})/image: "
                f"{infer_output['inference_time']:4f}"
            )

            # Define result directory
            out_path = self.results_path / div_dir.parts[-1]
            out_path.mkdir(parents=True, exist_ok=True)

            self.save_metrics(infer_output, out_path)

            if save_plots:
                self.plot_predictions(infer_output, out_path)
