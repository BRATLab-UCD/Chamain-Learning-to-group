#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:07:58 2021
train.py to train CIFAR-100 with grouping
@author: chamain.gamage
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
import argparse

import math

from models import MCRR2CIFAR100Uniform

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--ngroups', default=1, type=int, help='number of groups')
parser.add_argument('--b', default=4000, type=int, help='batch size')
parser.add_argument('--lambda_', default=0.375, type=float, help='number of groups')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0

nepochs = 200
hidden_size = 12 #h in the paper

lambda_ =args.lambda_
batch_size= args.b
learning_rate = args.lr
subClasses= args.ngroups
clip = 0.5

#for further control of classification and rate
DClassLossT = 0.04
Rt = 0.01



labelLayers=2

lr2=0.1
lr3=0.0001





SummaryW = True
modelTag = 'example-cifar100'

comment = 'Grouping_Rt{}-sub_{}: model_{}-hidden_{}-batch_{}-lr_{}_{}_{}-lambda_{}'\
    .format(str(Rt),str(subClasses),str(modelTag),str(hidden_size),str(batch_size)\
            ,str(learning_rate),str(lr2),str(lr3),str(lambda_))
        
print(comment)

if(SummaryW):
    writer = SummaryWriter(comment=comment)
else:
    writer = None

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=4)



# Model
print('==> Building model..')
Encoder, Decoder,CodeRate,Channel,Labeller = MCRR2CIFAR100Uniform(hidden_size=hidden_size,device = device\
                                                       ,batchsize=batch_size,num_classes=100,layer_out=True\
                                              ,trainableC=True,initialZ=0,d=0\
                                                  ,num_groups=subClasses,layers=labelLayers,labeller_inplanes=2*2*hidden_size)

Encoder = Encoder.to(device)
Decoder = Decoder.to(device)
CodeRate = CodeRate.to(device)
Labeller = Labeller.to(device)
Channel = Channel.to(device)


## initialization for strpsize
nn.init.constant_(Channel.sroot_c,0.0395)



criterion = nn.CrossEntropyLoss()
pamsList1  = list(Encoder.parameters())+  list(Decoder.parameters()) + list(CodeRate.parameters()) 
pamsList2 = list(Labeller.parameters())
pamsList3 = list(Channel.parameters()) 

optimizer = optim.SGD([{'params':pamsList1},
                       {'params':pamsList2,'lr':lr2},
                       {'params':pamsList3,'lr':lr3}
                       ],lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)

    
scheduler = MultiStepLR(optimizer, milestones=[100,150,180], gamma=0.1)
#scheduler = MultiStepLR(optimizer, milestones=[10,20,30], gamma=0.1) #for pre-trained models


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    Encoder.train()
    Decoder.train()
    CodeRate.train()
    Labeller.train()
    Channel.train()
    train_loss = 0
    correct = 0
    total = 0           
    closs=0
    BPPD=0
    BPPC= 0
    deltaR=0
    epsS2=0
    epsS2exp=0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        ## 3-7 animal, else non animal
        
        #print('types:',targets.type(),subtargets.type(),targets[:10],subtargets[:10],)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        outs,z = Encoder(inputs)
        
        z_hat,c,eps2 = Channel(z,8,8)
        
        
        
        subtargets = Labeller(outs)
       
        DeltaR,Dloss,Closs,Pi = CodeRate(z,subtargets,eps2,subClasses,detachEps=False)
        
        
        #print(bppdeltaR,cdloss)
        bppD = Dloss/(inputs.shape[2]*inputs.shape[3])
        bppC = Closs/(inputs.shape[2]*inputs.shape[3])
        bppdeltaR = DeltaR/(inputs.shape[2]*inputs.shape[3])
        
       
        output_z_hat = Decoder(z_hat)
       
        
        cl_loss_z_hat = criterion(output_z_hat, targets)/math.log(2)
       
        Dcl_loss = torch.nn.functional.relu(cl_loss_z_hat-DClassLossT)
        bpploss = torch.nn.functional.relu(bppC-Rt)
        
        loss =  lambda_*bpploss + (1-lambda_)*Dcl_loss
        loss = loss.mean()
       
            
        # Backprpagation and optimization
            
        loss.backward()
        ## grad clipping
        clip_grad_norm_(pamsList1, clip)
        clip_grad_norm_(pamsList2, clip)
        clip_grad_norm_(pamsList2, clip)
        
        
        optimizer.step()
        train_loss += loss.item()
        closs += cl_loss_z_hat.mean().item()
        epsS2 += eps2.mean().item()
        epsS2exp += c.mean().item()
       
        deltaR += (bppdeltaR.mean()).item()
        BPPD += (bppD.mean()).item()
        BPPC += (bppC.mean()).item()
      
        _, predicted = output_z_hat.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        #print(epz)
    
        #print(Pi)
        print(batch_idx, len(trainloader), 'TLoss: %.3f | Acc: %.3f%% (%d/%d) | CLoss: %.3f\
              | DeltaR2: %.3f | BPP_Rd2: %.4f | BPP_Rc2: %.4f | eps22: %.6f | eps2exp: %.6f'
              % (train_loss/(batch_idx+1), 100.*correct/total, correct\
                , total,closs/(batch_idx+1),deltaR/(batch_idx+1),BPPD/(batch_idx+1)\
                ,BPPC/(batch_idx+1),epsS2/(batch_idx+1),epsS2exp/(batch_idx+1)))
            
    if(SummaryW):
        writer.add_scalar('Train/Total_loss',train_loss/len(trainloader),epoch+1)
        writer.add_scalar('Train/C_loss',closs/len(trainloader),epoch+1)
        writer.add_scalar('Train/Acc',100.*correct/total,epoch+1)
        writer.add_scalar('Train/DeltaR',deltaR/len(trainloader),epoch+1)
        writer.add_scalar('Train/BPPD',BPPD/len(trainloader),epoch+1)
        writer.add_scalar('Train/BPPC',BPPC/len(trainloader),epoch+1)
        writer.add_scalar('Train/Esp2',epsS2/len(trainloader),epoch+1)
        

def test(epoch):
    global best_acc
    Encoder.eval()
    Decoder.eval()
    CodeRate.eval()
    Labeller.eval()
    Channel.eval()
    test_loss = 0
    correct = 0
    total = 0
    BPPD=0
    BPPC= 0
    deltaR=0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
           
            inputs, targets = inputs.to(device), targets.to(device)
            outs,z = Encoder(inputs)
            z_hat,c,eps = Channel(z,8,8)
            subtargets = Labeller(outs)
            #bppdeltaR,Dloss,Closs,_ = MCR22(z,subtargets,eps,subClasses)
            bppdeltaR,Dloss,Closs,_ = CodeRate(z,subtargets,eps,subClasses)
            bppD = Dloss/(inputs.shape[2]*inputs.shape[3])
            bppC = Closs/(inputs.shape[2]*inputs.shape[3])
            bppdeltaR = bppdeltaR/(inputs.shape[2]*inputs.shape[3])
            #info_loss = torch.mean(kld)/math.log(2)
            #iloss += info_loss
            outputs = Decoder(z_hat)
            cl_loss = criterion(outputs, targets)/math.log(2)
            test_loss +=  cl_loss #+ beta*info_loss
            #IZY += math.log(10, 2) - cl_loss.item()## H(Y) - H(Y/Z)
            deltaR += bppdeltaR.mean().item()
            BPPD += bppD.mean().item()
            BPPC += bppC.mean().item()
           
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | DeltaR: %.3f | BPP_Rd: %.4f | BPP_Rc: %.4f '
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total,
                        deltaR/(len(testloader)), BPPD/(len(testloader)),BPPC/(len(testloader))))
        if(SummaryW):
            writer.add_scalar('Test/Total_loss',test_loss/len(testloader),epoch+1)
            writer.add_scalar('Test/Acc',100.*correct/total,epoch+1)
            writer.add_scalar('Test/DeltaR',deltaR/len(testloader),epoch+1)
            writer.add_scalar('Test/BPPD',BPPD/len(testloader),epoch+1)
            writer.add_scalar('Test/BPPC',BPPC/len(testloader),epoch+1)
           

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'Encoder': Encoder.state_dict(),
            'Decoder':Decoder.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpointCIFAR'):
            os.mkdir('checkpointCIFAR')
        if(SummaryW):
            torch.save(state, './checkpointCIFAR/'+comment+'_Bckpt.pth')
        best_acc = acc
    if epoch == (nepochs-1):
        print('Saving..')
        state = {
            'Encoder': Encoder.state_dict(),
            'Decoder':Decoder.state_dict(),
            'MCRR':CodeRate.state_dict(),
            'Labelling':Labeller.state_dict(),
            'Channel':Channel.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpointCIFAR'):
            os.mkdir('checkpointCIFAR')
        if(SummaryW):
            torch.save(state, './checkpointCIFAR/'+comment+'_Fckpt.pth')
        best_acc = acc
    

for i in range(start_epoch):
    scheduler.step()
for epoch in range(start_epoch, nepochs):
    train(epoch)
    test(epoch)
    scheduler.step()
if(SummaryW):
    writer.close()
