# CytoSegment

**CytoSegment** is a Python library for **semantic segmentation** of cytometry image data using deep learning. It is designed to support researchers and practitioners working with microscopic biological imagery—especially blood cells—by providing reliable tools for training, evaluating, and deploying segmentation models.

---

## Features

* Deep learning-based semantic segmentation
* Easy-to-use CLI for local or HPC training
* Supports multi-run experiments with parameter sweeps
* Optimized for cytometry and microorganism image datasets

---

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/RaghavaAlajangi/cytosegment.git
cd cytosegment
```

2. **Upgrade core Python packaging tools**:

```bash
python -m pip install --upgrade pip setuptools wheel
```

3. **Install the package**:

**Option 1: Standard installation**

```bash
pip install .
```

**Option 2: Editable/development mode**

```bash
pip install -e .
```

---

## Dataset Structure

Organize your dataset as follows:

```
dataset_root/
├── training/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── ...
│   └── masks/
│       ├── mask1.png
│       └── ...
└── testing/
    ├── images/
    │   ├── image1.jpg
    │   └── ...
    └── masks/
        ├── mask1.png
        └── ...
```

---

## Usage

Run training jobs with simple CLI commands:

#### View CLI options

```bash
cytosegment -h
```

#### Run a single training session

```bash
cytosegment data.path="path/to/dataset_root"
```

#### Customize training (example: learning rate)

```bash
cytosegment train.optimizer.learn_rate=0.01
```

#### Perform multiple runs with different learning rates

```bash
cytosegment -m train.optimizer.learn_rate=0.01,0.02,0.03
```

#### Run multi-experiments on HPC with SLURM

```bash
cytosegment -m slurm=true train.optimizer.learn_rate=0.01,0.02,0.03
```

---

## Documentation

Full documentation (with training workflows, architecture, and evaluation metrics) is coming soon.

---

## Contributing

Feel free to open issues or submit pull requests. Contributions are welcome!

---

## License

GPL-3.0 license
