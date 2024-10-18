import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml

from .base import BaseTracker
from ..models import summary
from ..tools import rename_ckp_path_with_md5


class LocalTracker(BaseTracker):
    def __init__(self, exp_dir, min_ckp_acc):
        super().__init__()
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.min_ckp_acc = min_ckp_acc

        # Create a folder to save model checkpoints
        self.ckp_dir = self.exp_dir / "checkpoints"
        self.ckp_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {}
        self.params = {}
        self.best_model_path = None

    @staticmethod
    def save_jit_model(model, metadata, jit_path):
        """Save a PyTorch model in TorchScript format along with metadata."""
        # Save torch JIT model
        model_scripted = torch.jit.script(model)
        model_scripted.save(jit_path)

        extra_files = {"meta": str(metadata)}
        torch.jit.save(model_scripted, jit_path, _extra_files=extra_files)

    @staticmethod
    def save_onnx_model(model, img_size, onnx_path):
        """Save a PyTorch model in ONNX format."""
        batch_size = 8
        # Input to the model
        dummy_input = torch.randn(batch_size, 1, img_size[0], img_size[1],
                                  requires_grad=True)
        torch.onnx.export(
            model.cpu(),  # model being run
            # model input (or a tuple for multiple inputs)
            dummy_input,
            # where to save the model (can be a file-like object)
            str(onnx_path),
            # store the trained  weights inside the model file
            export_params=True,
            # the ONNX version to export the model to
            opset_version=11,
            # whether to execute constant folding for optimization
            do_constant_folding=True,
            # the model's input names
            input_names=["input"],
            # the model's output names
            output_names=["output"],
            # variable length axes
            dynamic_axes={"input": {0: "batch_size"},
                          "output": {0: "batch_size"}}
        )

    def dump_logs(self):
        """Save training logs as a yaml file."""
        logs = {
            "params": self.params,
            "metrics": self.metrics
        }
        with open(self.exp_dir / "train_logs.yaml", "w") as f:
            yaml.dump(logs, f, default_flow_style=False, sort_keys=False)

    def log_param(self, name, param):
        self.params[name] = param

    def log_metric(self, name, metric):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(metric)

    def log_artifact(self, artifact_name, artifact_path):
        save_path = self.exp_dir / (artifact_name or artifact_path.name)
        save_path.write_bytes(Path(artifact_path).read_bytes())

    def log_model(self, model, new_ckp_name, metadata, img_shape):
        """Logs the model by saving it in JIT and ONNX formats."""
        # Make sure model is in eval mode
        model = model.eval()

        # Save JIT model
        jit_path = self.ckp_dir / f"{new_ckp_name}.ckp"
        self.save_jit_model(model, metadata, jit_path)

        # Compute and add md5sum to the model ckp path
        md5_ckp_path = rename_ckp_path_with_md5(jit_path)

        # Save ONNX model
        onnx_path = self.ckp_dir / f"{new_ckp_name}.onnx"
        self.save_onnx_model(model, img_shape, onnx_path)

        # Track best model path (for inferencing)
        self.best_model_path = md5_ckp_path

        # Compute and add md5sum to the model onnx path
        rename_ckp_path_with_md5(onnx_path)

    def log_model_summary(self, model, input_shape):
        result, params_info = summary(model, input_shape)

        lines = result.split("\n")
        # Open the CSV file in write mode
        with open(self.exp_dir / "model_summary.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for line in lines:
                writer.writerow([line])

    def plot_logs(self, save_path):
        """Plot training metrics."""
        epochs = np.arange(1, self.params["epochs"] + 1)
        ckp_data = self.metrics.get("ckp_flags", [])
        early_stop_epoch = self.metrics.get("early_stop", [])
        if ckp_data:
            ckp_epochs = np.array(ckp_data)[:, 0]
            ckp_val_acc = np.array(ckp_data)[:, 1]
            ckp_val_loss = np.array(ckp_data)[:, 2]
        else:
            ckp_epochs, ckp_val_acc, ckp_val_loss = [], [], []

        plt.figure(figsize=(14, 7), facecolor=(1, 1, 1))
        plt.rc("grid", linestyle="--", color="lightgrey")
        plt.subplot(121)
        plt.plot(epochs, self.metrics["train_acc"], color="red",
                 label="Train accuracy", zorder=1)
        plt.plot(epochs, self.metrics["val_acc"], color="green",
                 label="Valid accuracy", zorder=2)
        if len(ckp_epochs) > 1:
            plt.scatter(ckp_epochs, ckp_val_acc, c="blue", s=20, marker="x",
                        label=f"Saved_ckp(>={self.min_ckp_acc:.2f})", zorder=3)
        plt.legend(loc="lower right")
        plt.title("Accuracy plot")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.ylim(0.3, 1.1)
        plt.grid()
        plt.subplot(122)
        plt.plot(epochs, self.metrics["train_loss"], color="red",
                 label="Train loss", zorder=1)
        plt.plot(epochs, self.metrics["val_loss"], color="green",
                 label="Valid loss", zorder=2)
        if len(early_stop_epoch) == 1:
            plt.scatter(early_stop_epoch, ckp_val_loss, c="blue", s=20,
                        marker="x", label="Early_stopping", zorder=3)
        plt.legend(loc="upper right")
        plt.title("Loss plot")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.ylim(-0.01, 0.5)
        plt.grid()
        plt.savefig(save_path)

    def end_run(self):
        """Terminate tracker instance."""
        # Save training logs
        self.dump_logs()
        # Save training curves
        plot_path = self.exp_dir / "train_plot.png"
        self.plot_logs(plot_path)
        print(f"Logs saved in {self.exp_dir}")
