# Introduction
This repository is an implementation of the paper [ForensicsForest Family: A Series of Multi-scale Hierarchical Cascade Forests for Detecting GAN-generated Faces](https://arxiv.longhoe.net/abs/2308.00964) presented in TIFS 2024. This paper describes ForensicsForest Family, a novel set of forest-based methods to detect GAN-generate faces. In contrast to the recent efforts of using CNNs, we investigate the feasibility of using forest models and introduce a series of multi-scale hierarchical cascade forests, which are ForensicsForest, Hybrid ForensicsForest, and Divide-and-Conquer ForensicsForest, respectively. 

![image](/Overview.png)

# ForensicsForest

## **Requirements**

python 3.7

deep-forest

## Dataset Preparation

We include the dataset loaders for several commonly-used generated-faces datasets, i.e.,  [StyleGAN](https://github.com/NVlabs/stylegan) ,  [StyleGAN2](https://github.com/NVlabs/stylegan2) , and  [StyleGAN3](https://github.com/NVlabs/stylegan3) . You can enter the dataset website to download the original data. The folder structure is shown as follows.

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
To obtain the CSV file of the dataset, run the following script in your console.

`run get_csv.py`

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

This is an implementation of the paper [ForensicsForest Family: A Series of Multi-scale Hierarchical Cascade Forests for Detecting GAN-generated Faces](https://arxiv.longhoe.net/abs/2308.00964). 

This paper is extended from our ICME conference paper(Oral) [Forensics Forest: Multi-scale Hierarchical Cascade Forest for Detecting GAN-generated Faces](https://ieeexplore.ieee.org/abstract/document/10219895). 

If you find our ForensicsForest useful to your research, please cite it as follows:

```
@article{lu2024forensicsforest,
title={ForensicsForest Family: A Series of Multi-scale Hierarchical Cascade Forests for Detecting GAN-generated Faces},
author={Lu, Jiucui and Zhou, Jiaran and Li, Bin and Dong, Junyu and Lyu, Siwei and Li, Yuezun},
journal={IEEE Transactions on Information Forensics and Security},
year={2024}
}
```

```
@inproceedings{lu2023forensics,
  title={Forensics Forest: Multi-scale Hierarchical Cascade Forest for Detecting GAN-generated Faces},
  author={Lu, Jiucui and Li, Yuezun and Zhou, Jiaran and Li, Bin and Lyu, Siwei},
  booktitle={2023 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={2309--2314},
  year={2023},
  organization={IEEE}
}
```
