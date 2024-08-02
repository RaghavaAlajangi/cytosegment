# CytoSegment

This Python library is designed for the semantic segmentation of cytometry 
image data using advanced deep learning techniques. It is tailored to assist 
researchers and practitioners in the field of cytometry by providing 
tools for analyzing and segmenting microorganism images, such 
as blood cells, with high accuracy.

## Installation

#### Install in editable mode

1. Clone the project:  
```bash
git clone git@github.com:RaghavaAlajangi/cytosegment.git

or 

git clone https://github.com/RaghavaAlajangi/cytosegment.git
```

2. Go to the project directory:
```bash
cd cytosegment
```

3. Install the project in editable mode:  
```bash
pip install -e .
```

4. Usage:
```bash
# See the command line arguments:
cytosegment -h

# Train a model with a learning rate (single run)
cytosegment training.optimizer.learn_rate=0.01

# Train a model with different learning rates (multi run)
cytosegment -m training.optimizer.learn_rate=0.01,0.02,0.03  # with `-m` for multirun

# Train a model with different learning rates on hpc (multi run)
cytosegment -m slurm=true training.optimizer.learn_rate=0.01,0.02,0.03  # with `slurm=true` to run experiment on HPC
```