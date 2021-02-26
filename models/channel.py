#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:14:00 2021
Some parts are copied from MCR2
References
[1] Yu, Yaodong, et al. "Learning diverse and discriminative representations
 via the principle of maximal coding rate reduction." arXiv preprint arXiv:2006.08558 (2020).
@author: chamain.gamage
"""
import torch
import numpy as np
import torch.nn as nn

def label_to_membership(targets, num_classes=None):
    """Generate a true membership matrix, and assign value to current Pi.
    Parameters:
        targets (np.ndarray): matrix with one hot labels
    Return:
        Pi: membership matirx, shape (num_classes, num_samples, num_samples)
    """
    if(len(list(targets.shape))==1):
        targets = one_hot(targets, num_classes)
    num_samples, num_classes = targets.shape
    Pi = np.zeros(shape=(num_classes, num_samples, num_samples))
    for j in range(len(targets)):
        k = np.argmax(targets[j])
        Pi[k, j, j] = 1.
    return Pi

def label_to_membership_from_prob(targets, num_classes=None):
    """Generate a true membership matrix, and assign value to current Pi.
    Parameters:
        targets (np.ndarray): matrix with probs
    Return:
        Pi: membership matirx, shape (num_classes, num_samples, num_samples)
    """
    #targets = one_hot(targets, num_classes)
    num_samples, num_classes = targets.shape
    Pi = np.zeros(shape=(num_classes, num_samples, num_samples))
    for j in range(len(targets)):
        Pi[:, j, j] = targets[j]
    return Pi

def one_hot(labels_int, n_classes):
    """Turn labels into one hot vector of K classes. """
    labels_onehot = torch.zeros(size=(len(labels_int), n_classes)).float()
    for i, y in enumerate(labels_int):
        labels_onehot[i, y] = 1.
    return labels_onehot

class CodingRate(torch.nn.Module):
    def __init__(self, gam1=1.0, gam2=1.0, eps=0.01,d=0):
        super(CodingRate, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps
        self.d=d

    def compute_discrimn_loss_empirical(self, W,epsilon_sq,detachEps):
        """Empirical Discriminative Loss."""
        ## detach \epslon 
        if(detachEps):
            epsilon_sq = epsilon_sq.detach()
       
        p, m = W.shape
        I = torch.eye(p).cuda()
        scalar = p / (m * epsilon_sq)
        logdet = torch.logdet(I + self.gam1 * scalar * (W).matmul(W.T))
        return ((m+self.d)/m)*logdet / 2.

    def compute_compress_loss_empirical(self, W, Pi,epsilon_sq):
        """Empirical Compressive Loss."""
        p, m = W.shape
        #print('W shape:',W.shape,' Pi shape:',Pi.shape)
        k, _, _ = Pi.shape
        I = torch.eye(p).cuda()
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * epsilon_sq)
            log_det = torch.logdet(I + scalar * (W).matmul((Pi[j])).matmul(W.T))
            compress_loss += log_det * (trPi +self.d)/ m
        return compress_loss / 2.

   
    def cosine_loss(self,W,Pi):
        """Cosine loss: inter-group similarity - inner-group similarity"""
        totalSim = torch.sum(W.matmul(W.T))
        innerSum=0
        k, _, _ = Pi.shape
        for j in range(k):
            innerSum += torch.sum((Pi[j]).matmul(W).matmul(W.T).matmul(Pi[j]))
        interSim = totalSim - innerSum
        diffSim = torch.abs(interSim) - torch.abs(innerSum)
        #diffSim = torch.abs(interSim) - torch.relu(innerSum)
        return diffSim/W.size(0)
    
    def forward(self, X, Y, epsilon_sq,num_classes=None,overlapping=False,detachEps=False,cosineLoss=False):
        if num_classes is None:
            num_classes = Y.max() + 1
        W = X.T
        if(overlapping):
            Pi = label_to_membership_from_prob(Y.cpu().detach().numpy(), num_classes)
        else:
            Pi = label_to_membership(Y.cpu().detach().numpy(), num_classes)
        Pi = torch.tensor(Pi, dtype=torch.float32).cuda()

        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W,epsilon_sq,detachEps)
        compress_loss_empi = self.compute_compress_loss_empirical(W, Pi,epsilon_sq)
        cosine_loss = self.cosine_loss(W.T, Pi)
      
 
        total_loss_empi = self.gam2 * -discrimn_loss_empi + compress_loss_empi
      
        
        if(cosineLoss):
            return -total_loss_empi, discrimn_loss_empi, compress_loss_empi,Pi,cosine_loss
        else:
            return -total_loss_empi, discrimn_loss_empi, compress_loss_empi,Pi

## class for uniform channel
class ChannelUniform(nn.Module):
    def __init__(self,trainable,value,hidden_size,device,samples=True):
        super(ChannelUniform, self).__init__()
        self.trainable = trainable
        self.value = value
        self.hidden_size = hidden_size
        self.device = device
        #sqrt(c)
        self.sroot_c = nn.Parameter(torch.rand(1, dtype= torch.float,requires_grad=self.trainable).cuda())
        self.sroot_c.requires_grad=self.trainable
        self.samples=samples
    
    
    def forward(self,v,height,width):
        ## v= normalized out3
        if(self.samples):
            c = torch.square(self.sroot_c)
            averagedC = c/np.sqrt(v.size(1))
            #noise samples between [-c,c]
            nsample = torch.rand(v.size(0),height*width*self.hidden_size,device = self.device)*(2*averagedC)-averagedC
            
            ## quantization
            v_hat = torch.round(v/averagedC)*averagedC - v.detach() + v
            v_new = v+nsample
            
            #print('check shapes: V:',v.shape,' nsample:',nsample.shape)
            epsilon_sq = torch.mean(torch.sum(torch.square(v-v_new),1))
            epsilon_sqQuan = torch.mean(torch.sum(torch.square(v_new-v_hat),1))
            
            return v_hat.view(v.size(0),-1,height,width),epsilon_sqQuan,epsilon_sq
        else:
            return torch.mean(torch.square(torch.square(c)))