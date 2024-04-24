This is an implementation of the paper [ForensicsForest Family: A Series of Multi-scale Hierarchical Cascade Forests for Detecting GAN-generated Faces](https://ieeexplore.ieee.org/abstract/document/10219895)

If you find our ForensicsForest useful to your research, please cite it as follows:

```
@article{lu2023forensicsforest,
title={ForensicsForest Family: A Series of Multi-scale Hierarchical Cascade Forests for Detecting GAN-generated Faces},
author={Lu, Jiucui and Li, Yuezun and Zhou, Jiaran and Li, Bin and Dong, Junyu and Lyu, Siwei},
journal={arXiv preprint arXiv:2308.00964},
year={2023}
}
```

# Deep Forest

## **Requirements**

python 3.7

## **Quick Start**

Install the package *deep-forest* is available via PyPI.

`pip install deep-forest`

# ForensicsForest

## Folder Structure

```ruby
StyleGAN
|---train
    |---0
    |---1
    |---train.csv
|---test
    |---0
    |---1
    |---test.csv
```

Use the `get_csv.py` to obtain the CSV file of the dataset.

## Input Feature Extract
We use four scales as N = 1, 2, 3, 4. Then we can extract appearance and frequency features from N(N ≥ 1) patches of input images. Specially, we extract biology features from the whole
image. You can extract the input features for each scale using `extract_feature.py` (Our code uses N=4 as an example).

Note that `extract_feature.py` is only for the folder corresponding to the generated faces, you can modify the file path to extract the input features of the real faces, and then use `Merge.py` to concatenate the extracted features.
## Hierarchical Cascade Forest

For the `cascade.py` of package *deep-forest*, you can replace it with `ForensicsForest cascade.py`.

## Multi-scale Ensemble

Run `main2.py`, `main3.py` and `main4.py` respectively to obtain the augmented features of each sacle. Then run `main1.py` for final results.

# Hybrid ForensicsForest

For the `cascade.py` of package *deep-forest*, you can replace it with `Hybrid FF cascade.py`.

# Divide-and-Conquer ForensicsForest

For the `cascade.py` of package *deep-forest*, you can replace it with `D-and-C FF cascade.py`. For the `layer.py` of package *deep-forest*, you can replace it with `D-and-C FF layer.py`.
