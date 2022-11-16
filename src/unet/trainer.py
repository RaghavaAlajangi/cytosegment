import os
import json
import torch
import datetime
import numpy as np
import os.path as osp
from matplotlib import pyplot as plt
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import DataLoader

from early_stop_cb import EarlyStoppingCallback


class Trainer:
    def __init__(self,
                 model,
                 loss_function,
                 eval_metric,
                 opti_name,
                 scheduler_name,
                 data_dict,
                 batch_size=8,
                 use_cuda=False,
                 max_epochs=100,
                 learn_rate=0.001,
                 step_size=15,
                 out_folder='unet_results',
                 init_from_ckp=None,
                 experiment_name=None
                 ):
        self.model = model
        self.loss_function = loss_function
        self.eval_metric = eval_metric
        self.opti_name = opti_name
        self.scheduler_name = scheduler_name
        self.data_dict = data_dict
        self.batch_size = batch_size
        self.cuda = use_cuda
        self.max_epochs = max_epochs
        self.learn_rate = learn_rate
        self.step_size = step_size
        self.out_folder = out_folder
        self.experiment_name = 'test_run' if experiment_name is None \
            else experiment_name
        self.early_stop_callback = EarlyStoppingCallback(patience=7)
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        self.nowtime = datetime.datetime.now().strftime('%d-%m-%Y [%H.%M.%S]')
        self.model_params = round(
            sum(p.numel() for p in self.model.parameters()) / 1e6, 2)
        self.thresh_acc = 0.5

        if use_cuda:
            self.model = model.cuda()
            self.loss_function = loss_function.cuda()
            # self.model = DataParallel(model)

        self.optim = self.select_optimizer()
        self.scheduler = self.select_scheduler()
        self.data_loaders = self.data_loaders(self.data_dict)

        if init_from_ckp is not None:
            self.restore_checkpoint(init_from_ckp)

        if self.out_folder:
            os.makedirs(self.out_folder, exist_ok=True)

    def data_loaders(self, data_dict):
        data_load_dict = dict()
        for m in ['train', 'valid']:
            data_load_dict[m] = DataLoader(data_dict[m],
                                           batch_size=self.batch_size,
                                           # num_workers=8,
                                           shuffle=True)
        return data_load_dict

    def select_optimizer(self):
        if self.opti_name == 'Adam':
            return Adam(self.model.parameters(), lr=self.learn_rate)
        if self.opti_name == 'SGD':
            return SGD(self.model.parameters(), lr=self.learn_rate,
                       momentum=0.9)

    def select_scheduler(self):
        if self.scheduler_name == 'stepLR':
            return lr_scheduler.StepLR(optimizer=self.optim,
                                       step_size=self.step_size,
                                       gamma=0.1)
        if self.scheduler_name == 'ReduceLROnPlateau':
            return lr_scheduler.ReduceLROnPlateau(optimizer=self.optim,
                                                  mode='max',
                                                  factor=0.2)

    def restore_checkpoint(self, ckp_path):
        ckp = torch.load(ckp_path, map_location=self.device)
        self.model.load_state_dict(ckp['model_state_dict'])
        self.optim.load_state_dict(ckp['optimizer_state_dict'])
        for param in self.model.features.parameters():
            param.requires_grad = False

    def epoch_runner(self, mode):
        if mode == 'train':
            self.model.train()
        if mode == 'valid':
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
            with torch.set_grad_enabled(mode == 'train'):
                predicts = self.model(images)
                # [B, 1, H, W] --> [B, H, W] for loss calculation
                predicts = predicts.squeeze(1)
                loss = self.loss_function(predicts, labels)
                accuracy = self.eval_metric(predicts, labels)
                # Backward + optimize only in train mode
                if mode == 'train':
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
        if self.out_folder and not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder, exist_ok=True)
        with open(self.out_folder + '/train_logs.json', 'w') as fp:
            json.dump(logs, fp)

        epoch = logs['epochs']
        train_loss = logs['train_loss']
        train_acc = logs['train_acc']
        val_loss = logs['val_loss']
        val_acc = logs['val_acc']
        ckp_epochs = logs['ckp_flag']
        ckp_val_acc = [val_acc[epoch.index(ep)] for ep in ckp_epochs]

        plt.figure(figsize=(14, 7), facecolor=(1, 1, 1))
        plt.rc('grid', linestyle='--', color='lightgrey')
        plt.subplot(121)
        plt.scatter(ckp_epochs, ckp_val_acc, c='blue', s=20,
                    marker='x', label='Saved_ckp')
        plt.plot(epoch, train_acc, color='red', label="Train accuracy")
        plt.plot(epoch, val_acc, color='green', label="Valid accuracy")
        plt.legend(loc="lower right")
        plt.title('Accuracy plot')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.subplot(122)
        plt.plot(epoch, train_loss, color='red', label="Train loss")
        plt.plot(epoch, val_loss, color='green', label="Valid loss")
        plt.legend(loc="upper right")
        plt.title('Loss plot')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid()
        plt.savefig(self.out_folder + '/train_plot.png')
        plt.show()

    def print_train_logs(self, epoch_log):
        epoch, dynamic_lr, train_loss, train_acc, val_loss, val_acc = epoch_log
        print(f'[Epochs-{epoch}/{self.max_epochs} | lr:{dynamic_lr}]:')
        print(f'[Train_loss:{train_loss:.4f} | Train_acc:{train_acc:.4f} | '
              f'Val_loss:{val_loss:.4f} | Val_acc:{val_acc:.4f}]')

    def save_checkpoint(self, new_ckp, mode='jit'):
        if mode == 'jit':
            os.makedirs(osp.join(self.out_folder, 'torch_jit'), exist_ok=True)
            jit_path = self.out_folder + '/torch_jit/' + new_ckp + '_jit.ckp'
            model_scripted = torch.jit.script(self.model)
            model_scripted.save(jit_path)
        else:
            os.makedirs(osp.join(self.out_folder, 'torch_norm'), exist_ok=True)
            norm_path = self.out_folder + '/torch_norm/' + new_ckp + '_org.ckp'
            torch.save({'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optim.state_dict(),
                        }, norm_path)

    def train(self):
        val_acc_list = []
        train_logs = {'epochs': [],
                      'dynamicLR': [],
                      'train_loss': [],
                      'train_acc': [],
                      'val_loss': [],
                      'val_acc': [],
                      'ckp_flag': []
                      }
        for epoch in range(1, self.max_epochs + 1):
            train_loss, train_acc = self.epoch_runner(mode='train')
            val_loss, val_acc = self.epoch_runner(mode='valid')
            val_acc_list.append(val_acc)
            dynamic_lr = [group['lr'] for group in
                          self.optim.param_groups][0]
            train_logs['epochs'].append(epoch)
            train_logs['dynamicLR'].append(dynamic_lr)
            train_logs['train_loss'].append(train_loss)
            train_logs['train_acc'].append(train_acc)
            train_logs['val_loss'].append(val_loss)
            train_logs['val_acc'].append(val_acc)
            epoch_log = [epoch, dynamic_lr, train_loss,
                         train_acc, val_loss, val_acc]
            self.print_train_logs(epoch_log)

            if val_acc > self.thresh_acc and val_acc == max(val_acc_list):
                new_ckp = f'/E{epoch}_validAcc_{int(val_acc * 1e4)}'
                self.save_checkpoint(new_ckp)
                train_logs['ckp_flag'].append(epoch)

            if self.scheduler:
                self.scheduler.step()
            self.early_stop_callback.step(val_loss)
            if self.early_stop_callback.should_stop():
                break
        self.plot_save_logs(train_logs)
