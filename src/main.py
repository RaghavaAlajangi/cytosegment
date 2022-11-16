from pathlib import Path

from matplotlib import pyplot as plt

from unet.models import UNet
from unet.trainer import Trainer
from unet.dataset import UNetDataset, split_dataset
from unet.criterion import FocalTverskyLoss
from unet.eval_metrics import IoUCoeff

OPIMIZER_NAME = 'Adam'
SCHEDULER_NAME = 'stepLR'
VALID_SIZE = 0.2
BATCH_SIZE = 8
IN_CHANNELS = 1
OUT_CLASSES = 1
MAX_EPOCHS = 2
LEARN_RATE = 0.001
STEP_SIZE = 10
AUGMENT = False
# Mean and std values are calculated on benchmark_version3.hdf5
MEAN = [0.6735]
STD = [0.1342]
NUM_SAMPLES = 500
USE_CUDA = True
RESULTS = 'unet_test'
EXPERIMENT_NAME = 'unet_test'

# Loss function params
ALPHA = 0.3
BETA = 0.7
GAMMA = 0.75

hdf5_path = r'C:\Raghava_local\GITLAB\rtdc-segmentation\data\datasets' \
            r'\benchmark_version3.hdf5'
json_path = r'C:\Raghava_local\BENCHMARK_DATA\test'

json_files = [p for p in Path(json_path).rglob('*.json') if p.is_file()]

print(len(json_files))

# unet_dataset = UNetDataset.from_json_files(lbl_files, AUGMENT, MEAN, STD)
# dataDict = split_dataset(unet_dataset, VALID_SIZE)

unet_dataset = UNetDataset.from_hdf5_data(hdf5_path, AUGMENT, MEAN, STD)
dataDict = split_dataset(unet_dataset, VALID_SIZE)

print(dataDict['train'].__len__())
print(dataDict['valid'].__len__())

tds = dataDict['train']
for i in range(2):
    img, msk = tds.__getitem__(i)
    print('Image shape:', img.shape)
    print('Mask shape:', msk.shape)
    print('Image range: [{},{}]'.format(img.min(), img.max()))
    print('Mask range: [{},{}]'.format(msk.min(), msk.max()))

    plt.imshow(img.permute(1, 2, 0), 'gray')
    plt.show()
    plt.imshow(msk, 'gray')
    plt.show()

unet_model = UNet(n_channels=IN_CHANNELS, n_classes=OUT_CLASSES)
loss_function = FocalTverskyLoss(alpha=ALPHA, beta=BETA, gamma=GAMMA)
eval_metric = IoUCoeff()

trainer = Trainer(model=unet_model,
                  loss_function=loss_function,
                  eval_metric=eval_metric,
                  opti_name=OPIMIZER_NAME,
                  scheduler_name=SCHEDULER_NAME,
                  data_dict=dataDict,
                  batch_size=BATCH_SIZE,
                  use_cuda=USE_CUDA,
                  max_epochs=MAX_EPOCHS,
                  learn_rate=LEARN_RATE,
                  step_size=STEP_SIZE,
                  out_folder=RESULTS,
                  init_from_ckp=None,
                  experiment_name=EXPERIMENT_NAME
                  )

print('Started training.....')
trainer.train()
print('Finished training!')
