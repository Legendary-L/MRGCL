# MRGCL


Codes, datasets for paper "Multi-Relational Graph Contrastive Learning with Learnable Graph Augmentation"

![AppVeyor](https://img.shields.io/badge/python-3.9.13-blue)
![AppVeyor](https://img.shields.io/badge/numpy-1.23.5-red)
![AppVeyor](https://img.shields.io/badge/pytorch-2.0.0-brightgreen)
![AppVeyor](https://img.shields.io/badge/torch--geometric-2.0.0-orange)
![AppVeyor](https://img.shields.io/badge/scipy-1.10.1-purple)
![AppVeyor](https://img.shields.io/badge/matplotlib-3.8.1-brown)


## Run code

For how to use MRCGNN, we present an example based on the Deng's dataset.

1.Learning structural features from graphs, you need to change the path in 'drugfeature_fromMG.py' first. If you want use MRGCL on your own dataset, please ensure the datas in 'trimnet' folds and the datas in 'codes for MRGCL' folds are the same.)

```
python drugfeature_fromMG.py
```

2.Training/validating/testing for 5 times and get the average scores of multiple metrics.
```
python 5timesrun.py
```

## Dataset

Deng    |    Ryu    |    FB15K-237    |    WN18RR

