import datetime
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import DataLoader
import yaml

from .criterion import FocalTverskyLoss
from .dataset import UNetDataset, split_dataset
from .early_stopping import EarlyStopping
from .eval_metrics import IoUCoeff
from .models import UNet


class SetTrainer:
    def __init__(self,
                 model,
                 data_dict,
                 criterion,
                 eval_metric,
                 optimizer_name,
                 scheduler_name,
                 batch_size=8,
                 learn_rate=0.001,
                 max_epochs=100,
                 use_cuda=False,
                 lr_step_size=15,
                 min_ckp_acc=0.85,
                 early_stop_patience=10,
                 path_out="experiments",
                 num_workers=None,
                 init_from_ckp=None,
                 ):
        self.model = model
        self.data_dict = data_dict
        self.loss_function = criterion
        self.eval_metric = eval_metric
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.max_epochs = max_epochs
        self.cuda = use_cuda
        self.lr_step_size = lr_step_size
        self.path_out = Path(path_out)
        self.min_ckp_acc = min_ckp_acc
        self.num_workers = num_workers

        self.early_stop = EarlyStopping(early_stop_patience)
        self.device = torch.device("cuda" if self.cuda else "cpu")

        if use_cuda:
            self.model = model.cuda()
            self.criterion = criterion.cuda()
            # self.model = DataParallel(model)

        self.optim = self.select_optimizer()
        self.scheduler = self.select_scheduler()
        self.data_loaders = self.data_loaders(self.data_dict)

        if init_from_ckp is not None:
            self.restore_checkpoint(init_from_ckp)

        # Create a folder to store experiment results
        nowtime = datetime.datetime.now().strftime('%d-%m-%Y [%H.%M.%S]')
        self.nowtime_path = self.path_out / nowtime
        self.nowtime_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def with_params(cls, params_file_path):
        # Load params file (.yaml)
        params = yaml.safe_load(open(params_file_path))

        model_params = params["model"]
        in_channels = model_params.get("in_channels")
        out_classes = model_params.get("out_classes")
        model = UNet(in_channels, out_classes)

        data_params = params["data"]
        data_type = data_params["type"]
        data_path = data_params.get("data_path")
        augmentation = data_params.get("augmentation")
        valid_size = data_params.get("valid_size")
        batch_size = data_params.get("batch_size")
        mean = data_params.get("mean")
        std = data_params.get("std")
        num_workers = data_params.get("num_workers")
        if data_type == "JSON":
            unet_dataset = UNetDataset.from_json_files(data_path, augmentation,
                                                       mean, std)
        else:
            unet_dataset = UNetDataset.from_hdf5_data(data_path, augmentation,
                                                      mean, std)

        data_dict = split_dataset(unet_dataset, valid_size)

        criterion_params = params["criterion"]
        alpha = criterion_params.get("alpha")
        beta = criterion_params.get("beta")
        gamma = criterion_params.get("gamma")
        criterion = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)

        # metric_params = params["metric"]
        metric = IoUCoeff()

        optimizer_name = params["optimizer"]["type"]

        scheduler_params = params["scheduler"]
        scheduler_name = scheduler_params["type"]
        lr_step_size = scheduler_params["lr_step_size"]

        learn_rate = params["learn_rate"]
        max_epochs = params["max_epochs"]
        use_cuda = params["use_cuda"]
        min_ckp_acc = params["min_ckp_acc"]
        early_stop_patience = params["early_stop_patience"]
        path_out = params["path_out"]
        init_from_ckp = params["init_from_ckp"]

        return cls(model, data_dict, criterion, metric, optimizer_name,
                   scheduler_name, batch_size, learn_rate, max_epochs,
                   use_cuda, lr_step_size, min_ckp_acc, early_stop_patience,
                   path_out, num_workers, init_from_ckp)

    def data_loaders(self, data_dict):
        data_load_dict = dict()
        for m in data_dict.keys():
            data_load_dict[m] = DataLoader(data_dict[m],
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=True)
        return data_load_dict

    def select_optimizer(self):
        if self.optimizer_name == "Adam":
            return Adam(self.model.parameters(), lr=self.learn_rate)
        if self.optimizer_name == "SGD":
            return SGD(self.model.parameters(), lr=self.learn_rate,
                       momentum=0.9)

    def select_scheduler(self):
        if self.scheduler_name == "stepLR":
            return lr_scheduler.StepLR(optimizer=self.optim,
                                       step_size=self.lr_step_size,
                                       gamma=0.1)
        if self.scheduler_name == "ReduceLROnPlateau":
            return lr_scheduler.ReduceLROnPlateau(optimizer=self.optim,
                                                  mode="max",
                                                  factor=0.2)

    def restore_checkpoint(self, ckp_path):
        ckp = torch.load(ckp_path, map_location=self.device)
        self.model.load_state_dict(ckp["model_state_dict"])
        self.optim.load_state_dict(ckp["optimizer_state_dict"])
        for param in self.model.features.parameters():
            param.requires_grad = False

    def epoch_runner(self, mode):
        if mode == "train":
            self.model.train()
        if mode == "valid":
            self.model.eval()
        loss_list = []
        acc_list = []
        # Get batch of images and labels iteratively
        for images, labels in self.data_loaders[mode]:
            images = images.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.float32)
            # Zero the parameter gradients
            self.optim.zero_grad()
            # Track history only in train mode
            with torch.set_grad_enabled(mode == "train"):
                predicts = self.model(images)
                # [B, 1, H, W] --> [B, H, W] for loss calculation
                predicts = predicts.squeeze(1)
                loss = self.criterion(predicts, labels)
                accuracy = self.eval_metric(predicts, labels)
                # Backward + optimize only in train mode
                if mode == "train":
                    loss.backward()
                    self.optim.step()
                loss_item = loss.item()
                del loss  # this may be the fix for my OOM error
                loss_list.append(loss_item)
                acc_list.append(accuracy)
        loss_avg = float(np.stack(loss_list).mean())
        acc_avg = float(np.concatenate(acc_list).mean())
        return loss_avg, acc_avg

    def plot_save_logs(self, logs):
        with open(self.nowtime_path / "train_logs.json", "w") as fp:
            json.dump(logs, fp)

        epoch = logs["epochs"]
        train_loss = logs["train_loss"]
        train_acc = logs["train_acc"]
        val_loss = logs["val_loss"]
        val_acc = logs["val_acc"]
        ckp_epochs = logs["ckp_flag"]
        ckp_val_acc = [val_acc[epoch.index(ep)] for ep in ckp_epochs]

        plt.figure(figsize=(14, 7), facecolor=(1, 1, 1))
        plt.rc("grid", linestyle="--", color="lightgrey")
        plt.subplot(121)
        plt.scatter(ckp_epochs, ckp_val_acc, c="blue", s=20,
                    marker="x", label="Saved_ckp")
        plt.plot(epoch, train_acc, color="red", label="Train accuracy")
        plt.plot(epoch, val_acc, color="green", label="Valid accuracy")
        plt.legend(loc="lower right")
        plt.title("Accuracy plot")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.subplot(122)
        plt.plot(epoch, train_loss, color="red", label="Train loss")
        plt.plot(epoch, val_loss, color="green", label="Valid loss")
        plt.legend(loc="upper right")
        plt.title("Loss plot")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid()
        plt.savefig(self.nowtime_path / "train_plot.png")
        plt.show()

    def print_train_logs(self, epoch_log):
        epoch, dynamic_lr, train_loss, train_acc, val_loss, val_acc = epoch_log
        print(f"[Epochs-{epoch}/{self.max_epochs} | lr:{dynamic_lr}]:")
        print(f"[Train_loss:{train_loss:.4f} | Train_acc:{train_acc:.4f} | "
              f"Val_loss:{val_loss:.4f} | Val_acc:{val_acc:.4f}]")

    def save_checkpoint(self, new_ckp, mode="jit"):
        if mode == "jit":
            jit_fold = self.nowtime_path / "torch_jit"
            jit_fold.mkdir(parents=True, exist_ok=True)
            jit_path = jit_fold / "torch_jit" / new_ckp + "_jit.ckp"
            model_scripted = torch.jit.script(self.model)
            model_scripted.save(jit_path)
        else:
            torch_fold = self.nowtime_path / "torch_norm"
            torch_fold.mkdir(parents=True, exist_ok=True)
            org_path = torch_fold / "torch_norm" / new_ckp + "_org.ckp"
            torch.save({"model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        }, org_path)

    def train(self):
        val_acc_list = []
        train_logs = {"epochs": [],
                      "dynamicLR": [],
                      "train_loss": [],
                      "train_acc": [],
                      "val_loss": [],
                      "val_acc": [],
                      "ckp_flag": []
                      }
        for epoch in range(1, self.max_epochs + 1):
            train_loss, train_acc = self.epoch_runner(mode="train")
            val_loss, val_acc = self.epoch_runner(mode="valid")
            val_acc_list.append(val_acc)
            dynamic_lr = [group["lr"] for group in
                          self.optim.param_groups][0]
            train_logs["epochs"].append(epoch)
            train_logs["dynamicLR"].append(dynamic_lr)
            train_logs["train_loss"].append(train_loss)
            train_logs["train_acc"].append(train_acc)
            train_logs["val_loss"].append(val_loss)
            train_logs["val_acc"].append(val_acc)
            epoch_log = [epoch, dynamic_lr, train_loss,
                         train_acc, val_loss, val_acc]
            self.print_train_logs(epoch_log)

            if val_acc > self.min_ckp_acc and val_acc == max(val_acc_list):
                new_ckp = f"/E{epoch}_validAcc_{int(val_acc * 1e4)}"
                self.save_checkpoint(new_ckp)
                train_logs["ckp_flag"].append(epoch)

            if self.scheduler:
                self.scheduler.step()
            self.early_stop(val_loss)
            if self.early_stop.should_stop:
                print("Early stopping!")
                break
        self.plot_save_logs(train_logs)
