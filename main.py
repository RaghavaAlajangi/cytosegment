import time
from datetime import timedelta

from matplotlib import pyplot as plt

from unet.models import UNet
from unet.trainer import SetTrainer
from unet.dataset import UNetDataset, split_dataset
from unet.criterion import FocalTverskyLoss
from unet.metrics import IoUCoeff

# model
IN_CHANNELS = 1
OUT_CLASSES = 1

# Data
VALID_SIZE = 0.2
BATCH_SIZE = 8
AUGMENT = False
# Mean and std values are calculated based
# on benchmark_version3.hdf5
# MEAN = [0.6735]
# STD = [0.1342]

MEAN = [0.6795]
STD = [0.1417]

NUM_WORKERS = 0

# Optimizer
OPIMIZER_NAME = "Adam"

# Scheduler
SCHEDULER_NAME = "stepLR"
LR_STEP_SIZE = 10

# Loss function
ALPHA = 0.3
BETA = 0.7
GAMMA = 0.75

# Other hyper-parameters
MAX_EPOCHS = 5
LEARN_RATE = 0.001
EARLY_STOP_PATIENCE = 10
MIN_CKP_ACC = 0.5

NUM_SAMPLES = 200  # can be used for testing
USE_CUDA = True
PATH_OUT = "experiments"

hdf5_path = "data/segm_dataset.hdf5"
json_path = r"C:\Raghava_local\BENCHMARK_DATA\test"

dataset_mode = "HDF5"

if dataset_mode == "JSON":
    unet_dataset = UNetDataset.from_json_files(json_path, AUGMENT, MEAN, STD)
else:
    unet_dataset = UNetDataset.from_hdf5_data(hdf5_path, AUGMENT, MEAN, STD,
                                              num_samples=NUM_SAMPLES)

dataDict = split_dataset(unet_dataset, VALID_SIZE)

print(dataDict["train"].__len__())
print(dataDict["valid"].__len__())

tds = dataDict["train"]
for i in range(2):
    img, msk = tds.__getitem__(i)
    print("Image shape:", img.shape)
    print("Mask shape:", msk.shape)
    print("Image range: [{},{}]".format(img.min(), img.max()))
    print("Mask range: [{},{}]".format(msk.min(), msk.max()))

    plt.imshow(img.permute(1, 2, 0), "gray")
    plt.show()
    plt.imshow(msk, "gray")
    plt.show()

unet_model = UNet(n_channels=IN_CHANNELS, n_classes=OUT_CLASSES)
criterion = FocalTverskyLoss(alpha=ALPHA, beta=BETA, gamma=GAMMA)
eval_metric = IoUCoeff()

trainer = SetTrainer(model=unet_model,
                     data_dict=dataDict,
                     criterion=criterion,
                     eval_metric=eval_metric,
                     optimizer_name=OPIMIZER_NAME,
                     scheduler_name=SCHEDULER_NAME,
                     batch_size=BATCH_SIZE,
                     learn_rate=LEARN_RATE,
                     max_epochs=MAX_EPOCHS,
                     use_cuda=USE_CUDA,
                     lr_step_size=LR_STEP_SIZE,
                     min_ckp_acc=MIN_CKP_ACC,
                     early_stop_patience=EARLY_STOP_PATIENCE,
                     path_out=PATH_OUT,
                     num_workers=NUM_WORKERS,
                     init_from_ckp=None
                     )

tik = time.time()
print("Started training.....")
trainer.train()
tok = time.time() - tik
train_time = str(timedelta(seconds=tok)).split('.')[0]
print(f"Total training time: {train_time}")
