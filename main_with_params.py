import time
from datetime import timedelta

from unet.trainer import SetTrainer

params_path = "params/train_params_unet.yaml"

trainer = SetTrainer.with_params(params_path)

tik = time.time()
print("Started training.....")
trainer.train()
tok = time.time() - tik
train_time = str(timedelta(seconds=tok)).split('.')[0]
print(f"Total training time: {train_time}")
