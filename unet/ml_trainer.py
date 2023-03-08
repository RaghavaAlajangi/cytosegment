import datetime
import json
import shutil
from pathlib import Path
import time

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml

from .early_stopping import EarlyStopping
from .ml_criterions import get_criterion_with_params
from .ml_dataset import get_dataloaders_with_params
from .ml_metrics import get_metric_with_params
from .ml_models import get_model_with_params
from .ml_optimizers import get_optimizer_with_params
from .ml_schedulers import get_scheduler_with_params


class SetupTrainer:
    def __init__(self,
                 model,
                 dataloaders,
                 criterion,
                 metric,
                 optimizer,
                 scheduler,
                 max_epochs=100,
                 use_cuda=False,
                 min_ckp_acc=0.85,
                 early_stop_patience=10,
                 path_out="experiments",
                 tensorboard=False,
                 init_from_ckp=None
                 ):
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.metric = metric
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.use_cuda = use_cuda
        self.min_ckp_acc = min_ckp_acc
        self.tensorboard = tensorboard

        self.early_stop = EarlyStopping(early_stop_patience)
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if self.use_cuda:
            self.model = model.cuda()
            self.criterion = criterion.cuda()
            # self.model = DataParallel(model)

        # Print and save No of model parameters in the result logs
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Trainable parameters in the model:", trainable_params)
        self.model_params = trainable_params

        if init_from_ckp is not None:
            self.restore_checkpoint(init_from_ckp)

        if Path(path_out).is_dir():
            self.exp_path = Path(path_out)
        # if path_out == "experiments":
        else:
            # Create a folder to store experiment results
            time_now = datetime.datetime.now().strftime('%d_%b_%Y_%H%M%S%f')
            self.exp_path = Path(path_out) / time_now
            self.exp_path.mkdir(parents=True, exist_ok=True)

        if self.tensorboard:
            self.writer = SummaryWriter(str(self.exp_path / "tensorboard/"))

    @classmethod
    def with_params(cls, params_file_path):
        # Load params file (.yaml)
        params = yaml.safe_load(open(params_file_path))

        model = get_model_with_params(params)
        dataloaders = get_dataloaders_with_params(params)
        criterion = get_criterion_with_params(params)
        metric = get_metric_with_params(params)
        optimizer = get_optimizer_with_params(params, model)
        scheduler = get_scheduler_with_params(params, optimizer)

        max_epochs = params.get("max_epochs")
        use_cuda = params.get("use_cuda")
        min_ckp_acc = params.get("min_ckp_acc")
        early_stop_patience = params.get("early_stop_patience")
        path_out = params.get("path_out")
        init_from_ckp = params.get("init_from_ckp")
        tensorboard = params.get("tensorboard")

        # Create a folder to store experiment results based on current time
        time_now = datetime.datetime.now().strftime('%d_%b_%Y_%H%M%S%f')
        exp_path = Path(path_out) / time_now
        exp_path.mkdir(parents=True, exist_ok=True)

        # Copy experiment params.yaml file in results folder
        out_file_path = exp_path / Path(params_file_path).name

        # Copy the file from source to destination
        shutil.copy(params_file_path, out_file_path)

        return cls(model, dataloaders, criterion, metric, optimizer,
                   scheduler, max_epochs, use_cuda, min_ckp_acc,
                   early_stop_patience, str(exp_path), tensorboard,
                   init_from_ckp)

    def restore_checkpoint(self, ckp_path):
        ckp = torch.load(ckp_path, map_location=self.device)
        self.model.load_state_dict(ckp["model_state_dict"])
        self.optimizer.load_state_dict(ckp["optimizer_state_dict"])
        for param in self.model.features.parameters():
            param.requires_grad = False

    def epoch_runner(self, mode):
        if mode.lower() == 'train':
            self.model.train()
        if mode.lower() == 'valid':
            self.model.eval()
        loss_list = []
        predict_list = []
        target_list = []
        # Get batch of images and labels iteratively
        for images, labels in self.dataloaders[mode]:
            images = images.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.float32)
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            # Track history only in train mode
            with torch.set_grad_enabled(mode.lower() == 'train'):
                predicts = self.model(images)
                # [B, 1, H, W] --> [B, H, W] for loss calculation
                predicts = predicts.squeeze(1)
                loss = self.criterion(predicts, labels)
                # Backward + optimize only in train mode
                if mode.lower() == 'train':
                    loss.backward()
                    self.optimizer.step()
                loss_item = loss.item()
                del loss  # this may be the fix for my OOM error
                loss_list.append(loss_item)

                predict_list.append(predicts)
                target_list.append(labels)

        predict_torch = torch.cat(predict_list, dim=0)
        target_torch = torch.cat(target_list, dim=0)

        scores = self.metric(predict_torch, target_torch)
        loss_avg = float(np.stack(loss_list).mean())
        acc_avg = float(scores.mean())
        return loss_avg, acc_avg

    def plot_save_logs(self, logs):
        with open(self.exp_path / "train_logs.json", "w") as fp:
            json.dump(logs, fp, indent=1)

        epoch = logs["epochs"]
        train_loss = logs["train_loss"]
        train_acc = logs["train_acc"]
        val_loss = logs["val_loss"]
        val_acc = logs["val_acc"]
        ckp_epochs = logs["ckp_flag"]
        early_stop_epoch = logs["early_stop"]
        ckp_val_acc = [val_acc[epoch.index(ep)] for ep in ckp_epochs]
        ckp_val_loss = [val_loss[epoch.index(ep)] for ep in early_stop_epoch]

        plt.figure(figsize=(14, 7), facecolor=(1, 1, 1))
        plt.rc("grid", linestyle="--", color="lightgrey")
        plt.subplot(121)
        plt.plot(epoch, train_acc, color="red", label="Train accuracy")
        plt.plot(epoch, val_acc, color="green", label="Valid accuracy")
        plt.scatter(ckp_epochs, ckp_val_acc, c="blue", s=20,
                    marker="x", label="Saved_ckp")
        plt.legend(loc="lower right")
        plt.title("Accuracy plot")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.ylim(0.6, 1.0)
        plt.grid()
        plt.subplot(122)
        plt.plot(epoch, train_loss, color="red", label="Train loss")
        plt.plot(epoch, val_loss, color="green", label="Valid loss")
        plt.scatter(early_stop_epoch, ckp_val_loss, c="blue", s=20,
                    marker="x", label="Early_stopping")
        plt.legend(loc="upper right")
        plt.title("Loss plot")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.ylim(0.0, 0.5)
        plt.grid()
        plt.savefig(self.exp_path / "train_plot.png")
        # plt.show()

    def print_train_logs(self, epoch_log):
        epoch, dynamic_lr, train_loss, train_acc, val_loss, val_acc = epoch_log
        print(f"[Epochs-{epoch}/{self.max_epochs} | lr:{dynamic_lr}]:")
        print(f"[Train_loss:{train_loss:.4f} | Train_acc:{train_acc:.4f} | "
              f"Val_loss:{val_loss:.4f} | Val_acc:{val_acc:.4f}]")

    def save_checkpoint(self, new_ckp_name, mode="jit"):
        if mode == "jit":
            jit_fold = self.exp_path / "torch_jit"
            jit_fold.mkdir(parents=True, exist_ok=True)
            jit_path = str(jit_fold) + f"/{new_ckp_name}_jit.ckp"
            model_scripted = torch.jit.script(self.model)
            model_scripted.save(jit_path)
        else:
            torch_fold = self.exp_path / "torch_original"
            torch_fold.mkdir(parents=True, exist_ok=True)
            org_path = str(torch_fold) + f"/{new_ckp_name}_org.ckp"
            torch.save({"model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        }, org_path)

    def start_train(self):
        start_time = time.time()
        val_avg_acc_list = []
        train_logs = {"model_params": self.model_params,
                      "train_samples": len(self.dataloaders["train"].dataset),
                      "valid_samples": len(self.dataloaders["valid"].dataset),
                      "training_time": 0,
                      "epochs": [],
                      "dynamicLR": [],
                      "train_loss": [],
                      "train_acc": [],
                      "val_loss": [],
                      "val_acc": [],
                      "ckp_flag": [],
                      "early_stop": []
                      }
        for epoch in range(1, self.max_epochs + 1):
            train_avg_loss, train_avg_acc = self.epoch_runner(mode="train")
            val_avg_loss, val_avg_acc = self.epoch_runner(mode="valid")
            val_avg_acc_list.append(val_avg_acc)
            dynamic_lr = [group["lr"] for group in
                          self.optimizer.param_groups][0]
            train_logs["epochs"].append(epoch)
            train_logs["dynamicLR"].append(dynamic_lr)
            train_logs["train_loss"].append(train_avg_loss)
            train_logs["train_acc"].append(train_avg_acc)
            train_logs["val_loss"].append(val_avg_loss)
            train_logs["val_acc"].append(val_avg_acc)
            epoch_log = [epoch, dynamic_lr, train_avg_loss,
                         train_avg_acc, val_avg_loss, val_avg_acc]
            self.print_train_logs(epoch_log)

            # Tensorboard tracking loss and accuracies
            if self.tensorboard:
                self.writer.add_scalar("Training Loss", train_avg_loss, epoch)
                self.writer.add_scalar("Training Acc", train_avg_acc, epoch)
                self.writer.add_scalar("Validation Loss", val_avg_loss, epoch)
                self.writer.add_scalar("Validation Acc", val_avg_acc, epoch)

            if val_avg_acc > self.min_ckp_acc and val_avg_acc == max(
                    val_avg_acc_list):
                new_ckp_name = f"/E{epoch}_trainAcc_" \
                               f"{int(train_avg_acc * 1e4)}_validAcc_" \
                               f"{int(val_avg_acc * 1e4)}"
                self.save_checkpoint(new_ckp_name)
                train_logs["ckp_flag"].append(epoch)

            if self.scheduler:
                self.scheduler.step()
            self.early_stop(val_avg_loss)
            if self.early_stop.should_stop:
                train_logs["early_stop"].append(epoch)
                print("Early stopping!")
                break
        # Calculate training time and save it in results logs
        end_time = time.time() - start_time
        train_time = str(datetime.timedelta(seconds=end_time)).split('.')[0]
        train_logs["training_time"] = train_time
        # Plot and save results logs
        self.plot_save_logs(train_logs)

    def close(self):
        if self.tensorboard:
            self.writer.flush()
            self.writer.close()
