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



## Data perparation

1. Create JSON files (`labelme`) using `rtdc_to_json.py` script through CLI
    ````
   unet/rtdc_to_json.py --path_in <rtdc file> --path_out <optional> --bb_ckp_path <optional> --unet_ckp_path <optional> -k ml_score_r1f=10 -c
   ````


