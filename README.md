This is an implementation of the paper [ForensicsForest Family: A Series of Multi-scale Hierarchical Cascade Forests for Detecting GAN-generated Faces](https://ieeexplore.ieee.org/abstract/document/10219895)

If you find our ForensicsForest useful to your research, please cite it as follows:

# ForensicsForest

## **Requirements**

python 3.7

## **Quick Start**

Install the package *deep-forest* is available via PyPI.

`pip install deep-forest`
## Dataset Structure
```ruby
StyleGAN
|---0_fake
|---1_real
```
## Input Feature Extract
We use four scales as N = 1, 2, 3, 4. Then we can extract appearance and frequency features from N(N ≥ 1) patches of input images. Specially, we extract biology features from the whole
image. You can extract the input features for each scale using `extract_feature.py` (Our code uses N=4 as an example).

## Hierarchical Cascade Forest

## Multi-scale Ensemble

# Hybrid ForensicsForest

# Divide-and-Conquer ForensicsForest
