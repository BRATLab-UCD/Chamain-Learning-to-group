#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:14:15 2021
Labelling class to predict group labels
@author: chamain.gamage
"""
import torch.nn as nn
import torch.nn.functional as F

class Labelling(nn.Module):
    def __init__(self,num_classes=10,layers=2,inplanes=2*2*16,avpool=4):
        super(Labelling, self).__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(inplanes, num_classes)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn1 = nn.BatchNorm2d(16)
        self.avpool=avpool
        self.layers=layers
    
    def forward(self, x):
        out0 = x[0]
        out1 = x[1]
        out2 = x[2]
        out3 = x[3]
        
        if(self.layers==1):
            out = out2
            out = F.avg_pool2d(out, self.avpool*2)
        elif(self.layers==2):
            out= out3
            out = F.avg_pool2d(out, self.avpool)
        
        out = out.view(out.size(0), -1)
        #print('before linear:',out.shape)
        out = self.linear(out)
        out = F.softmax(out,1)
        return out
