# ForensicsForest

This is an implementation of the paper [ForensicsForest Family: A Series of Multi-scale Hierarchical Cascade Forests for Detecting GAN-generated Faces](https://ieeexplore.ieee.org/abstract/document/10219895)

## **Requirements**

python 3.7

## **Quick Start**

Install the package *deep-forest* is available via PyPI.

`pip install deep-forest`

## Input Feature Extract
We use four scales as N = 1, 2, 3, 4. Then we can extract appearance and frequency features from N(N ≥ 1) patches of input images. Specially, we extract biology features from the whole
image. You can extract the appearance and frequency features for each scale using `N=1.py`, `N=2.py`, `N=3.py`, `N=4.py`, and extract biology features using `landmarks.py`.


## Hierarchical Cascade Forest

## Multi-scale Ensemble

# Hybrid ForensicsForest

# Divide-and-Conquer ForensicsForest
