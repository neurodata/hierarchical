from setuptools import setup, find_packages

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from tasksim import task_similarity

from graspy.embed import AdjacencySpectralEmbed as ASE
from graspy.cluster import AutoGMMCluster as GMM

from proglearn import LifelongClassificationForest as l2f
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from graspy.embed import ClassicalMDS as CMDS

import time
from tqdm import tqdm

from joblib import Parallel, delayed

from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import pairwise_distances

requirements = [
    "graspy",
    "proglearn",
    "sklearn",
    "numpy",
    "matplotlib",
    "seaborn",
]

with open("README.md", mode="r", encoding = "utf8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="tasksim",
    version="0.0.1",
    author="Hayden Helm",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/neurodata/hierarchical/",
#    license="MIT",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
#        "License :: OSI Approved :: MIT License",
#        "Programming Language :: Python :: 3",
#        "Programming Language :: Python :: 3.6",
#        "Programming Language :: Python :: 3.7"
    ],
    install_requires=requirements,
#    packages=find_packages(exclude=["tests", "tests.*", "tests/*"]),
#    include_package_data=True
)
