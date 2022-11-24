# SemanticSegmentor

In this repository, the UNet model (semantic segmentation) data preparation and training code are stored.

## Installation

1. Install git lfs to clone the model checkpoints:  
```bash
git lfs install
```

2. Clone the project:  
```bash
git clone git@gitlab.gwdg.de:blood_data_analysis/SemanticSegmentor.git
```
3. Go to the project directory:
```bash
cd SemanticSegmentor
```

3.Install the project in editable mode:  
```bash
pip install -e .
```

## Data perparation

We are using a semi-automated labeling process to create a dataset for UNet training 
which means, instead of creating segmented labels from scratch, we use classification 
and segmentation models that are being trained before as estimates to generate the initial 
version of the label, and then we edit manually if it still does not meet the requirements.

### Step-1:
The `.json` files (that can be opened in labelme-GUI) are produced with the help of bloody_bunny 
(new model with minmax normalization) and unet model predictions. Use `rtdc_to_json.py` script to do so. 
- To see the options:

```bash
python unet/rtdc_to_json.py --help
```
```bash
Options:
  --path_in FILE         Path to RTDC dataset (.rtdc)
  --path_out PATH        Path to save output files (images, JSON files). If it
                         is not given, script creates new folder based on
                         `path_in`
  --bb_ckp_path FILE     Path to bloody_bunny model checkpoint. If it is not
                         given default checkpoint is taken from model folder
  --unet_ckp_path FILE   Path to unet model checkpoint. If it is not given
                         default checkpoint is taken from model folder
  -s, --min_score FLOAT  Specify minimum probability of `ml_score` feature
  -m, --ml_feat_kv TEXT  KEY=VALUE argument for cell_type and num_samples
                         pairthat needs to be extracted from RTDC dataset.i.e
                         `ml_score_r1f=10` If it is not given,
                         `ml_score`features in dataset are used to generate
                         labelme samples(by default 50 samples from each type)
  -c, --is_cuda          Specify whether cuda device available or not
  --help                 Show this message and exit.
```
- To use:
```bash
python unet/rtdc_to_json.py --path_in "C:/ralajan/rtdc_data_1/test.rtdc" -s 0.8 -m ml_score_r1f=10 -c
```
### Step-2:
Manual editing  

### Step-3:

Convert `.json` files into `.hdf5` file that contains images and binary mask. 
Use `json_to_hdf5.py` script to do so.

```bash
python unet/json_to_hdf5.py --help
```
```bash
Options:
  --path_in PATH   Path to .json files folder
  --path_out FILE  Optional. Path to save output (.hdf5) file
  --help           Show this message and exit.
```
- To use:
```bash
python unet/json_to_hdf5.py --path_in "C:/ralajan/rtdc_data_1"
```

It is not necessary to create `.hdf5` file from `.json` files to train a model because we can train 
a model with `.hdf5` file as well as `.json` files.


# Model training

## With params file:

Change the required parameters in `unet_params.yaml` file and then run the below command for training

```bash
python train.py --params_path "params/unet_params.yaml"
```

During training, a folder (`experiments`) will be created to save the experiments. Moreover, 
sub-folders will be created based on date and time of the experiment, and model checkpoints 
(at different validation accuracies), train logs, and plots will be saved there. Checkpoint path 
consists of epoch number and validation accuracy like example below.
```bash
E28_validAcc_9081_jit.ckp > E{checkpoint number}_validAcc_{validation accuracy}_{type of model}.ckp
```

## Without params file:
Sometimes, we want to train a model in jupyter notebooks because:
- We can apply trial and error method to select hyper-parameters quickly
- To visually inspect data samples, how do they look? how do transforms look like? to see some stats (shapes, normalization, image ranges)
- To see the model convergence progress
- To debug

There is an example notebook in the repo `training_notebook.ipynb`. You can use it for training.

### How to create a model object?
```bash
from unet.ml_models import UNet
unet_model = UNet(n_channels=IN_CHANNELS, n_classes=OUT_CLASSES)
```
### How to use and create dataset object?

We can create dataset object with `json_files_path` as well as `hdf5_file_path` as shown below.
```bash
from unet.ml_dataset import UNetDataset

# With json_files_path
unet_dataset = UNetDataset.from_json_files(json_path, AUGMENT, MEAN, STD)

# With hdf5_file_path
unet_dataset = UNetDataset.from_hdf5_data(hdf5_path, AUGMENT, MEAN, STD)
```
Data splitting for training and validation. As you can see below, `data_dict` is a pathon dictionary 
contains dataset objects for both `training` and `validation`
```bash
from unet.ml_dataset import split_dataset

data_dict = split_dataset(unet_dataset, VALID_SIZE)
```

Finally, create dataloaders
```bash
from unet.ml_dataset import create_dataloaders

dataloaders = create_dataloaders(data_dict, BATCH_SIZE, NUM_WORKERS)
```

Instead, if you have all the parameters in `params.yaml` file, we can simply create dataloaders with 
`get_dataloaders_with_params()` function like below.

```bash
from unet.ml_dataset import get_dataloaders_with_params

dataloaders = get_dataloaders_with_params(params)
```