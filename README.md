# SemanticSegmentor

In this repository, the UNet model (semantic segmentation) data preparation and training code are stored.

## Aims of this project

- [ ] A command line interface:

  - that accepts an `.rtdc` file and creates `.json`
  - files that we edit manually in `labelme` software to generate clean dataset 
    for semantic segmentation training using `UNet`. 
  - that accepts a folder containing `.json` files and generates `.hdf5` file
    containing images and masks, which can be used for training.
  

- [ ]  The whole training scripts to train a model:

    - Model
    - Metric
    - Criterion
    - Dataset
    - Trainer

- [ ] Bash script to train a model on HPC
- [ ] Bash script to perdict a dataset using trained `UNet` (dcevent)

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
Use `rtdc_to_json.py` script to do so.

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

It is not necessary to create `.hdf5` file out of `.json` files to train a model. We can 
train model in 

## Creating Dataset object



## Model training

### With params

Change the required parameters in `train_params_unet.yaml` file and then run the below command

```bash
python main_with_params.py --params_path "params/train_params_unet.yaml"
```

Once the training is finished, a folder (`experiments`) will be created and model checkpoints 
(at different validation accuracies), train logs, and plots will be saved there.

### Without params


