#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:12:18 2021
contains groupnet models
@author: chamain.gamage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from .resNet import BasicBlockNoAct,BasicBlock
from .channel import CodingRate,ChannelUniform
from .labeller import Labelling

class Encoder(nn.Module):
    def __init__(self, block, hidden_size,num_blocks,device,batchsize=1000,num_classes=10,layer_out=False):
        super(Encoder, self).__init__()
        self.layer_out=layer_out
        self.in_planes = 16
        self.hidden_size = hidden_size
        self.device = device
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.batchsize = batchsize
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 16, num_blocks[1], stride=2)
        self.layer3_1 = self._make_layer(BasicBlockNoAct, self.hidden_size, num_blocks[2]-1, stride=2)
        
     

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out0 = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3_1(out2)
        
        height = out3.size(2)
        width = out3.size(3)
        v =(out3).view(out3.size(0), -1)
        
        v = F.normalize(v,dim=1) 
       
       
        if(self.layer_out):
            return [out0,out1,out2,out3],v
        else:
            return v
        
class Decoder(nn.Module):
    def __init__(self, block, hidden_size,num_blocks,num_classes=100):
        super(Decoder, self).__init__()
        self.in_planes = 128
        self.conv2 = nn.Conv2d(hidden_size, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer3_2 = self._make_layer(block, 128, 1, stride=1)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.linear = nn.Linear(256*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #out = F.relu(x)## can remove if necessary
        out = self.conv2(x)
        out = self.layer3_2(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #print('before linear:',out.shape)
        out = self.linear(out)
        return out
    

        out = out.view(out.size(0), -1)
        #print('before linear:',out.shape)
        out = self.linear(out)
        return out

def MCRR2CIFAR100Uniform2(hidden_size=256,device = 'cuda',batchsize=256,num_classes=100\
                          ,layer_out=False,trainableC=True,initialZ=1.0,d=0,num_groups=10,layers=2,labeller_inplanes=2*2*16):
    return Encoder(BasicBlock, hidden_size,[2, 2, 2, 2],device,batchsize,num_classes,layer_out=layer_out)\
        ,Decoder(BasicBlock,hidden_size,[2,2,2,2],num_classes)\
        , CodingRate(d=d)\
        , ChannelUniform(trainableC,initialZ,hidden_size,device)\
        , Labelling(num_groups,layers,labeller_inplanes)
        
        
        