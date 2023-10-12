# SemanticSegmentor

This repository contains the training code for the UNet model used in semantic 
segmentation.

## Installation

#### Install as a package
```bash
semanticsegmentor@git+ssh://git@gitlab.gwdg.de/blood_data_analysis/SemanticSegmentor.git
```

#### Install in editable mode

1. Clone the project:  
```bash
git clone git@gitlab.gwdg.de:blood_data_analysis/SemanticSegmentor.git
```

2. Go to the project directory:
```bash
cd SemanticSegmentor
```

3. Install the project in editable mode:  
```bash
pip install -e .
```

4. Usage:
```bash
python -m semanticsegmentor --params_path <params.yaml file>
```



# Model training

## Locally:

Change the required parameters in `params/unet_params.yaml` file and then run the below command for training

Note: Please note that in this file, you have the option to define a list of 
parameters for each entry, allowing the generation of all conceivable combinations.

```bash
model:
  type: [UNetTunable]
  in_channels: [1]
  out_classes: [1]
  conv_block: [double]
  depth: [2]
  filters: [3]
  dilation: [1]
  batch_norm: [True]
  dropout: [0.1]                # Options: in the range of [0.1, 0.9]
  up_mode: [upconv]             # Options: upconv, upsample
  attention: [False]
  relu: [True]
  weight_init: [HeNormal]       # Options: default, xavier, HeUniform, orthogonal

dataset:
  type: [PNG]
  data_path: [data/training_testing_set_only_enrichment.zip]
  # "min_max" = True: minimum and maximum pixel values of each image will be
  # used to normalize the dataset. Otherwise, 255 will be used to divide each
  # and every image in the dataset
  img_size: [[64, 256]]
  min_max: [False]
  augmentation: [True]
  valid_size: [0.15]
  batch_size: [8]

  # Mean and std values for the whole dataset should be computed before
  # starting the training using 'unet/dataset_utils/compute_mean_std.py'
  # script. The computed values can be used during the model inference.
  # If there is a change in the dataset (either inclusion or exclusion of
  # image and mask pairs into the dataset), we need to recalculate the mean
  # and std values of the dataset again.
  mean: [0.48748]
  std: [0.08355]

  random_seed: [42]
  # how many subprocesses to use for data loading. 0 means that the data
  # will be loaded in the main process.
  num_workers: [0]

criterion:
  type: [FocalTverskyLoss]
  alpha: [0.5] # weightage: alpha --> FP, (1-alpha) --> FN
  gamma: [1.5]

metric:
  type: [DiceCoeff]


optimizer:
  type: [Adam]
  learn_rate: [0.001]

# Decays the learning rate by ``lr_decay_rate`` every ``patience``
# epochs. Sometimes model might be ended up within local minima because
# of the high learning rate. A scheduler will help the model overcome
# this by minimizing the learning rate progressively.

# NOTE: instead of using step based scheduler, the ReduceLROnPlateau
# scheduler could be more effective to get optimal results.
scheduler:
  type: [ReduceLROnPlateau]
  patience: [10]
  lr_decay_rate: [0.5] # Ex: after 5 epochs lr becomes 0.01*0.1 = 0.001

others:
  max_epochs: [200]
  use_cuda: [False]
  path_out: [test]

  # Trainer start saving checkpoints only after the validation
  # accuracy is higher than  'min_ckp_acc'
  min_ckp_acc: [0.96]

  # If the model metric (validation loss) starts increasing,
  # 'early stopping' will count the n consequent epochs (patience).
  # If still there is no improvement (after patience) training will
  # be terminated automatically. It saves computational power by
  # discarding the unnecessary epochs.
  # early_stop_patience: [15]

  # Start the training where you left by providing
  # previous checkpoint (not jit) path
  init_from_ckp: [null]

  # Specify whether results should be saved with tensorboard
  tensorboard: [False]

hpc_params:
  mail_id: raghava.alajangi@mpl.mpg.de
  max_mem_GB: 5
  max_time_hours: 1


```

```bash
# Train models locally
python job_handler.py --local
```



## On HPC:
1. Clone the repository on HPC (`login is required with access token`)
```bash
git clone git@gitlab.gwdg.de:blood_data_analysis/SemanticSegmentor.git
```
2. change required parameters in `params/unet_params.yaml`. Especially,
```bash
- num_workers -->  to 8
- use_cuda --> True
- hpc_params:
    mail_id --> mail ID
```

5. Run the jobs
```bash
python job_handler.py --slurm
```
6. Copy the experiments (trained models) from HPC to local drive
```bash
scp -r <USERNAME>@raven.mpcdf.mpg.de:/u/<USERNAME>/SemanticSegmentor/<exp path> <path to local directory>
```
