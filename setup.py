#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os.path import exists, dirname, realpath
import platform
from setuptools import setup, find_packages
import sys

maintainer = "Raghava Alajangi"
maintainer_email = "raghava.alajangi@mpl.mpg.de"
description = "Semantic Segmentation package for the LC project"
name = "unet"
year = "2022"

sys.path.insert(0, realpath(dirname(__file__)) + "/" + name)

extras_require = {
    "cpu": ["torch>=1.13.1",
            "torchvision>=0.13.1"],
    "gpu": ["torch>=1.13.1+cu117",
            "torchvision>=0.14.1"],
}

setup(
    name=name,
    maintainer=maintainer,
    maintainer_email=maintainer_email,
    url="https://gitlab.gwdg.de/blood_data_analysis/SemanticSegmentor",
    version="0.0.1",
    packages=find_packages(),
    package_dir={name: name},
    include_package_data=True,
    license="",
    description=description,
    long_description=open("README.md").read() if exists("README.md") else "",
    install_requires=[
        "albumentations",
        "click",
        "dclab",
        "h5py",
        "numpy",
        "matplotlib",
        "pyyaml",
        "scikit-image",
        "scipy",
    ],
    extras_require=extras_require,
    python_requires=">=3.9, <4",
    keywords=["RT-DC", "segmentation"],
    classifiers=["Operating System :: OS Independent",
                 "Programming Language :: Python :: 3",
                 "Topic :: Scientific/Engineering",
                 "Intended Audience :: Private",
                 ],
    platforms=["ALL"],
)
