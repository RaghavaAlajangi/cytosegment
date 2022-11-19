import time
from datetime import timedelta

from unet.trainer import SetTrainer
# from unet.dataset_utils import compute_mean_std


# hdf5_path = "data/segm_dataset.hdf5"
# mean, std = compute_mean_std(hdf5_path)
#
# print(mean, std)


params_path = "params/train_params_unet.yaml"

trainer = SetTrainer.with_params(params_path)

tik = time.time()
print("Started training.....")
trainer.train()
tok = time.time() - tik
train_time = str(timedelta(seconds=tok)).split('.')[0]
print(f"Total training time: {train_time}")
