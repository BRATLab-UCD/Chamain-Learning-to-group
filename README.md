# Learning-to-group
This repo contains the codes for the paper **'An End-to-End Learning Architecture for Efficient Image Encoding and Deep Learning'** presented at EUSIPCO-2021 [paper](https://ieeexplore.ieee.org/iel7/9615915/9615917/09616312.pdf) - An End-to-End Learning Architecture for Efficient Image Encoding and Deep Learning


## Overview
<img src="https://github.com/chamain/Learning-to-group/blob/main/images/groupMain2.png" width="500">

The proposed framework consists of an Encoder, a set of coding profiles and a task model (in this case, an image classifier). Encoder maps the input image  to a low-dimensional latent vector and generates a group label out of *k* groups. We call the group label *j* as the **profile index**. Coding profiles set contains *k* number of different encoding-decoding profiles that are optimized to compress the latent vectors of the corresponding profile index *j*. One such profile may typically feature a quantizer, an entropy coder, an entropy decoder and a de-quantizer (optional). Classier takes the decoded latent vector as the input and classify it in to one of *c* classification classes.

## Results
These are the results on CIFAR-100 data set.

<img src="https://github.com/chamain/Learning-to-group/blob/main/images/cifar100Uniformd12.png" width="500">

The following is the results for ImageNet-1k (32x32) data set.

<img src="https://github.com/chamain/Learning-to-group/blob/main/images/imageNetAcc5Uniform.png" width="500">

## Training
```
python train.py
```
## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{chamain2021end,
  title={An End-to-End Learning Architecture for Efficient Image Encoding and Deep Learning},
  author={Chamain, Lahiru D and Qi, Siyu and Ding, Zhi},
  booktitle={2021 29th European Signal Processing Conference (EUSIPCO)},
  pages={691--695},
  year={2021},
  organization={IEEE}
}
```
