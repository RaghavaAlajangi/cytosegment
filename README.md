# CytoSegment


[![Pipeline](https://github.com/RaghavaAlajangi/cytosegment/actions/workflows/ci.yml/badge.svg)](https://github.com/RaghavaAlajangi/cytosegment/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/RaghavaAlajangi/cytosegment/branch/main/graph/badge.svg?token=Z4FAPNDJWN)](https://codecov.io/gh/RaghavaAlajangi/cytosegment)
![GitHub License](https://img.shields.io/github/license/RaghavaAlajangi/cytosegment)
[![DOI](https://zenodo.org/badge/DOI/10.1002/cyto.a.24947.svg)](https://onlinelibrary.wiley.com/doi/10.1002/cyto.a.24947)


**CytoSegment** is a Python library for **semantic segmentation** of cytometry image data using deep learning. It is designed to support researchers and practitioners working with microscopic biological imagery—especially blood cells—by providing reliable tools for training, evaluating, and deploying segmentation models.

---

## 🚀 Features

* Deep learning-based semantic segmentation
* Easy-to-use CLI for local or HPC training
* Supports multi-run experiments with parameter sweeps
* Optimized for cytometry and microorganism image datasets


## 📄 Citation

If you use **cytosegment** in your research or work, please cite our paper:

> **Small U-Net for Fast and Reliable Segmentation in Imaging Flow Cytometry**  
> *Sara Kaliman, Raghava Alajangi, Nadia Sbaa, Paul Müller, Nadine Ströhlein, Jeffrey Harmon, Martin Kräter, Jochen Guck, Shada Abuhattum*  
> *Cytometry Part A (Wiley), 2025.*  
> DOI: [https://onlinelibrary.wiley.com/doi/10.1002/cyto.a.24947](https://onlinelibrary.wiley.com/doi/10.1002/cyto.a.24947)

**BibTeX:**
```bibtex
@article{https://doi.org/10.1002/cyto.a.24947,
author = {Kaliman, Sara and Alajangi, Raghava and Sbaa, Nadia and Müller, Paul and Ströhlein, Nadine and Harmon, Jeffrey and Kräter, Martin and Guck, Jochen and Abuhattum, Shada},
title = {Small U-Net for Fast and Reliable Segmentation in Imaging Flow Cytometry},
journal = {Cytometry Part A},
volume = {107},
number = {7},
pages = {450-463},
keywords = {cell features, deformability cytometry (DC), high-speed imaging, high-throughput, imaging flow cytometry, lab on a chip (LoC), segmentation, semantic segmentation, small U-net, U-netartificial intelligence},
doi = {https://doi.org/10.1002/cyto.a.24947},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/cyto.a.24947},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/cyto.a.24947},
abstract = {ABSTRACT Imaging flow cytometry requires rapid and accurate segmentation methods to ensure high-quality cellular morphology analysis and cell counting. In deformability cytometry (DC), a specific type of imaging flow cytometry, accurately detecting cell contours is critical for evaluating mechanical properties that serve as disease markers. Traditional thresholding methods, commonly used for their speed in high-throughput applications, often struggle with low-contrast images, leading to inaccuracies in detecting the object contour. Conversely, standard neural network approaches like U-Net, though effective in medical imaging, are less suitable for high-speed imaging applications due to long inference times. To address these issues, we present a solution that enables both fast and accurate segmentation, designed for imaging flow cytometry. Our method employs a small U-Net model trained on high-quality, curated, and annotated data. This optimized model outperforms traditional thresholding methods and other neural networks, delivering a 35× speed improvement on CPU over the standard U-Net. The enhanced performance is demonstrated by a significant reduction in systematic measurement errors in blood samples analyzed using DC. The tools developed in this study are adaptable for various imaging flow cytometry applications. This approach improves segmentation quality while maintaining the rapid processing necessary for high-throughput environments.},
year = {2025}
}

```

## 📦 Installation

```bash
# Clone repo
git clone https://github.com/RaghavaAlajangi/cytosegment.git
cd cytosegment

# Upgrade pip tools
python -m pip install --upgrade pip setuptools wheel

# Install (standard)
pip install .

# Or install in development mode
pip install -e .
```

## 🗂 Dataset Structure

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

## ⚡ Quick Start

Run training jobs with simple CLI commands:

```bash
# View CLI options
cytosegment -h

# Run a single training session
cytosegment data.path="path/to/dataset_root"

# Customize learning rate
cytosegment train.optimizer.learn_rate=0.01

# Multi-run experiments
cytosegment -m train.optimizer.learn_rate=0.01,0.02,0.03

# Multi-run on HPC (SLURM)
cytosegment -m slurm=true train.optimizer.learn_rate=0.01,0.02,0.03
```

## 📚 Documentation

Paper: [Cytometry Part A (2025)](https://onlinelibrary.wiley.com/doi/10.1002/cyto.a.24947)  
Code: [Original GitHub Repo](https://github.com/RaghavaAlajangi/CytoSegment-PyTorch)

## 🤝 Contributing

Feel free to open issues or submit pull requests. Contributions are welcome!


## 📜 License

GPL-3.0 license
