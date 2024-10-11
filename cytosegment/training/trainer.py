import csv
from datetime import timedelta
from pathlib import Path
import time
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
                 early_stop_patience=None,
                 path_out="experiments"
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

        # Create EarlyStop instance only if it is specified
        if not early_stop_patience:
            self.early_stop = EarlyStopping(early_stop_patience)
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.exp_path = Path(path_out)

        self.tracker = LocalTracker(self.exp_path, min_ckp_acc)

        if self.use_cuda:
            self.model = model.cuda()
            self.criterion = criterion.cuda()
            # self.model = DataParallel(model)

        images, _ = next(iter(self.dataloaders["train"]))
        self.image_shape = images[0].shape
        self.tracker.log_model_summary(model, self.image_shape)

        print(f"Training samples: {len(self.dataloaders['train'].dataset)}")
        print(f"Validation samples: {len(self.dataloaders['valid'].dataset)}")

    @classmethod
    def with_params(cls, params):
        model = get_model(params)
        dataloaders = get_dataloaders(params)
        criterion = get_criterion(params)
        metric = get_metric(params)
        optimizer = get_optimizer(params, model)
        scheduler = get_scheduler(params, optimizer)

        return cls(model, dataloaders, criterion, metric, optimizer,
                   scheduler, params.max_epochs, params.use_cuda,
                   params.min_ckp_acc,
                   params.early_stop_patience,
                   params.path_out)

    def epoch_runner(self, mode):
        if mode.lower() == "train":
            self.model.train()
        if mode.lower() == "valid":
            self.model.eval()
        loss_list = []
        # predict_list = []
        # mask_list = []

        bscore = []
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

                # predict_list.append(predicts)
                # mask_list.append(masks)

                bwise_scores = self.metric(predicts, masks)
                bscore.append(bwise_scores.cpu())

        # predict_tensor = torch.cat(predict_list, dim=0)
        # mask_tensor = torch.cat(mask_list, dim=0)
        # scores = self.metric(predict_tensor, mask_tensor)
        # acc_avg = float(scores.mean())

        loss_avg = float(np.stack(loss_list).mean())
        acc_avg = float(torch.mean(torch.stack(bscore)))

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

        # Get the data sample shape
        image_shape = self.dataloaders["train"].dataset.target_shape

        # Define model metadata
        metadata = {
            "image_shape": image_shape,
            "mean": self.dataloaders["train"].dataset.mean,
            "std": self.dataloaders["train"].dataset.std,
            "padding_ufunc": "np.mean"
        }

        # Log sample counts
        for mode in ["train", "valid", "test"]:
            self.tracker.log_param(f"{mode}_samples",
                                   len(self.dataloaders[mode].dataset))
        self.tracker.log_param("epochs", self.max_epochs)

        start_time = time.time()
        print("=" * 84)
        for epoch in range(1, self.max_epochs + 1):
            train_avg_loss, train_avg_acc = self.epoch_runner(mode="train")
            valid_avg_loss, valid_avg_acc = self.epoch_runner(mode="valid")
            dynamic_lr = [pg["lr"] for pg in self.optimizer.param_groups][0]
            self.tracker.log_metric("dynamicLR", dynamic_lr)
            self.tracker.log_metric("train_loss", train_avg_loss)
            self.tracker.log_metric("train_acc", train_avg_acc)
            self.tracker.log_metric("val_loss", valid_avg_loss)
            self.tracker.log_metric("val_acc", valid_avg_acc)

            print(f"|Epochs-{epoch}/{self.max_epochs} | lr:{dynamic_lr}|:")
            print(f"|Train_Loss:{train_avg_loss:.4f} | "
                  f"Train_Accuracy:{train_avg_acc:.4f} | "
                  f"Valid_Loss:{valid_avg_loss:.4f} | "
                  f"Valid_Accuracy:{valid_avg_acc:.4f}|")

            # Reduce the learning rate, if validation accuracy is not improving
            if type(self.scheduler).__name__ == "StepLR":
                self.scheduler.step()
            else:
                self.scheduler.step(valid_avg_loss)

            if valid_avg_acc > self.min_ckp_acc:

                # Record best model and its metrics
                if valid_avg_acc > best_valid_acc and epoch != self.max_epochs:
                    best_model = self.model
                    best_epoch = epoch
                    best_train_acc = train_avg_acc
                    best_valid_acc = valid_avg_acc
                    best_valid_loss = valid_avg_loss
                    # Reset flag when new best model is found
                    model_saved = False

                # Save the best model only when validation accuracy is
                # decreasing and model has not been saved yet
                if valid_avg_acc < best_valid_acc and not model_saved:
                    model_name = self.get_model_name(best_epoch,
                                                     best_train_acc,
                                                     best_valid_acc)

                    self.tracker.log_model(best_model, model_name, metadata,
                                           image_shape)
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
        print("=" * 84)
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
