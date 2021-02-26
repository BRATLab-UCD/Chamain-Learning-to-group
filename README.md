# Learning-to-group
This repo contains the codes for the paper `A Hierarchical Learning Architecture for End-to-End Image Compression and Classification' submitterd to EUSIPCO-2021

## Overview
1. This project uses CIFAR-10 dataset and we provide the python code for the end to end training of QuanNet in **keras**.
2. We use **ResNet-20** as the classification network
3. We use the following simple trainable layer for QuanNet.
%![alt text][logo]

%[logo]: https://github.com/chamain/QuanNet/blob/master/images/quanblock.PNG "Quan block"

## Results
These are the results on CIFAR-100 data set

The following is the results for ImageNet-1k (32x32) data set
## Training
```
python train.py
```
## Citation
