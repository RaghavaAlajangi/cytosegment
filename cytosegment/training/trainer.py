import csv
from datetime import timedelta
from pathlib import Path
import time

import numpy as np
import torch

from .early_stopping import EarlyStopping
from .criterions import get_criterion
from ..dataset import get_dataloaders
from .metrics import get_metric
from ..models import get_model
from .optimizers import get_optimizer
from .schedulers import get_scheduler
from ..ml_inferece import inference
from ..divided_group_inference import div_inference

from ..tracking import LocalTracker


class Trainer:
    def __init__(self, config):
        self.config = config

        model = get_model(config)
        self.dataloaders = get_dataloaders(config)
        self.criterion = get_criterion(config)
        self.metric = get_metric(config)
        self.optimizer = get_optimizer(config, model)
        self.scheduler = get_scheduler(config, self.optimizer)

        # Check if CUDA is available and assign the device
        self.device = torch.device(
            "cuda" if config.use_cuda and torch.cuda.is_available() else "cpu")
        print(f"Used device: {self.device}")

        # Create EarlyStop instance only if it is specified
        if not config.early_stop_patience:
            self.early_stop = EarlyStopping(config.early_stop_patience)

        self.exp_path = Path(config.path_out)

        self.tracker = LocalTracker(self.exp_path, config.min_ckp_acc)

        # Log model summary
        summary_size = config.data.img_size.copy()
        summary_size.insert(0, 1)
        self.tracker.log_model_summary(model, tuple(summary_size))

        # Move model and criterion to device
        self.model = model.to(self.device)
        self.criterion = self.criterion.to(self.device)

    def epoch_runner(self, mode):
        if mode.lower() == "train":
            self.model.train()
        if mode.lower() == "valid":
            self.model.eval()
        loss_list = []
        batch_scores = []

        # Get batch of images and labels iteratively
        for n, (images, masks) in enumerate(self.dataloaders[mode]):
            # Pass the data to the device
            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.float32)
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            # Track history only in train mode
            with torch.set_grad_enabled(mode.lower() == "train"):
                predicts = self.model(images)
                # [B, 1, H, W] --> [B, H, W] for loss calculation
                # predicts = predicts.squeeze(1)
                loss = self.criterion(predicts, masks)
                # Backward + optimize only in train mode
                if mode.lower() == "train":
                    loss.backward()
                    self.optimizer.step()
                loss_item = loss.item()
                del loss  # this may be the fix for my OOM error
                loss_list.append(loss_item)

                batch_score = self.metric(predicts, masks)
                batch_scores.append(batch_score.cpu())

        loss_avg = float(np.stack(loss_list).mean())
        acc_avg = float(torch.mean(torch.stack(batch_scores)))

        return loss_avg, acc_avg

    def dump_test_scores(self, scores):
        score_dict = {
            "img_path": scores[-1],
            "iou_scores": scores[1],
            "dice_scores": scores[2]
        }
        with open(self.exp_path / "test_scores.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(score_dict.keys())
            writer.writerows(zip(*score_dict.values()))

    @staticmethod
    def get_model_name(epoch, train_acc, valid_acc):
        return (f"E{epoch}_trainAcc_{int(train_acc * 1e4)}_"
                f"validAcc_{int(valid_acc * 1e4)}")

    def start_train(self):
        best_model = None
        best_epoch = 0
        best_train_acc = 0
        best_valid_acc = 0
        best_valid_loss = 0
        model_saved = False  # Flag to check if the best model has been saved

        # Define metadata that will be stored along with model checkpoint
        # (useful for deployment)
        metadata = {
            "img_size": self.config.data.img_size,
            "mean": self.dataloaders["train"].dataset.mean,
            "std": self.dataloaders["train"].dataset.std,
            "padding_ufunc": "np.mean"
        }

        # Log sample counts
        for mode in ["train", "valid", "test"]:
            self.tracker.log_param(f"{mode}_samples",
                                   len(self.dataloaders[mode].dataset))
            print(f"{mode.capitalize()} samples: "
                  f"{len(self.dataloaders[mode].dataset)}")
        # Log epochs
        self.tracker.log_param("epochs", self.config.max_epochs)

        start_time = time.time()
        print("=" * 87)
        for epoch in range(1, self.config.max_epochs + 1):
            train_avg_loss, train_avg_acc = self.epoch_runner(mode="train")
            valid_avg_loss, valid_avg_acc = self.epoch_runner(mode="valid")
            dynamic_lr = [pg["lr"] for pg in self.optimizer.param_groups][0]
            self.tracker.log_metric("dynamicLR", dynamic_lr)
            self.tracker.log_metric("train_loss", train_avg_loss)
            self.tracker.log_metric("train_acc", train_avg_acc)
            self.tracker.log_metric("val_loss", valid_avg_loss)
            self.tracker.log_metric("val_acc", valid_avg_acc)

            print(f"|Epochs-{epoch}/{self.config.max_epochs} | "
                  f"lr:{dynamic_lr}|:")
            print(f"|Train_Loss:{train_avg_loss:.4f} | "
                  f"Train_Accuracy:{train_avg_acc:.4f} | "
                  f"Valid_Loss:{valid_avg_loss:.4f} | "
                  f"Valid_Accuracy:{valid_avg_acc:.4f}|")

            # Reduce the learning rate, if validation accuracy is not improving
            if type(self.scheduler).__name__ == "StepLR":
                self.scheduler.step()
            else:
                self.scheduler.step(valid_avg_loss)

            # Is current validation accuracy better than defined accuracy.
            if valid_avg_acc > self.config.min_ckp_acc:
                # Record best model and its metrics
                if valid_avg_acc > best_valid_acc:
                    best_epoch = epoch
                    best_train_acc = train_avg_acc
                    best_valid_acc = valid_avg_acc
                    best_valid_loss = valid_avg_loss
                    # Reset flag when new best model is found
                    model_saved = False

                # Save the best model if:
                # 1. Validation accuracy is decreasing AND model has not been
                # saved yet OR
                # 2. It is the last epoch
                if (valid_avg_acc < best_valid_acc and not model_saved or
                        epoch == self.config.max_epochs):
                    model_name = self.get_model_name(best_epoch,
                                                     best_train_acc,
                                                     best_valid_acc)

                    self.tracker.log_model(best_model, model_name, metadata,
                                           self.config.data.img_size)
                    self.tracker.log_metric("ckp_flags",
                                            [best_epoch, best_valid_acc,
                                             best_valid_loss])

                    # Ensure model is only saved once until accuracy improves
                    model_saved = True

            # Record early stopping patience if it is specified
            if hasattr(self, "early_stop"):
                # Stop the training, if validation loss is not improving
                self.early_stop(valid_avg_loss)
                if self.early_stop.should_stop:
                    self.tracker.log_metric("early_stop", epoch)
                    print("Early stopping!")
                    break

        # Calculate training time and save it in results logs
        end_time = time.time() - start_time
        train_time = str(timedelta(seconds=end_time)).split(".")[0]
        self.tracker.log_param("training_time", train_time)

        # Reset the memory by deleting model and cache
        del self.model
        torch.cuda.empty_cache()
        print("=" * 87)
        print(f"Total training time: {train_time}")

        if self.tracker.best_model_path:
            test_results = inference(self.dataloaders["test"],
                                     self.tracker.best_model_path,
                                     self.exp_path,
                                     use_cuda=False,
                                     save_results=False)

            self.tracker.log_param("inference_cpu", test_results[0])
            self.tracker.log_param("test_iou_mean",
                                   float(test_results[1].mean()))
            self.tracker.log_param("test_dice_mean",
                                   float(test_results[2].mean()))

            self.dump_test_scores(test_results)

            # Testing divided groups with CPU device
            div_inference(self.tracker.best_model_path, self.exp_path,
                          use_cuda=False)

            # Run inference using gpu only it is available
            if torch.cuda.is_available():
                test_results = inference(self.dataloaders["test"],
                                         self.tracker.best_model_path,
                                         self.exp_path,
                                         use_cuda=True, save_results=False)
                self.tracker.log_param("inference_gpu", test_results[0])

                # Testing divided groups with CUDA device
                div_inference(self.tracker.best_model_path, self.exp_path,
                              use_cuda=True)

        # Plot and save results logs
        self.tracker.end_run()
