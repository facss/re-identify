#! /usr/bin/env python
#-*- coding:UTF-8 -*-
from __future__ import print_function
import os
import argparse
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from LoadData import ORLFaceData
from ContrastiveLoss import ContrastiveLoss
from Net import SiameseNet,OriNet #import three network
from Train import Siamese_Train
from Helper import func
from args import get_parser

#==================================================
myparser=get_parser()
opts=myparser.parse_args()
#==================================================

def main():

###################################### 1.Data Loader#################################################
##==============================Siamese network=========================================
    Siamese_traindata=ORLFaceData(opts.training_dir,transform=transforms.Compose([
        transforms.Scale((100,100)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([.5,.5,.5],[.5,.5,.5]),
    ]
    ),should_invert=False)

    Siamese_valdata=ORLFaceData(opts.validate_dir,transform=transforms.Compose([
        transforms.Scale((100,100)),
        transforms.ToTensor(),
        transforms.Normalize([.5,.5,.5],[.5,.5,.5]),
    ]),should_invert=False)

    Siamese_train_loader=DataLoader(Siamese_traindata,shuffle=True,num_workers=opts.num_workers,batch_size=opts.training_batch_size)
    Siamese_val_loader=DataLoader(Siamese_valdata,shuffle=True,num_workers=opts.num_workers,batch_size=opts.validate_batch_size)
##=============================Ori network=======================================
    Ori_traindata=ORLFaceData(opts.training_dir,transform=transforms.Compose([
        transforms.Scale((96,128)),
        transforms.RandomHorizontalFlip(),
    ]))

###################################### 2. Model ############################################################
##======================Siamese Network===========================
    opts.cuda=torch.cuda.is_available()
    if opts.cuda:
        SiameseNetModel=SiameseNet().cuda()
        torch.cuda.manual_seed(opts.seed)
    else :
        SiameseNetModel=SiameseNet()
        torch.manual_seed(opts.seed)

    criterion=ContrastiveLoss()
##=====================Ori Network===============================

####################################### 3.Optimizer ################################################################
    optimizer=optim.Adam(SiameseNetModel.parameters(),lr=opts.lr)
    if opts.resume:
        if os.path.isfile(opts.resume):
            print("=>loading checkpoint '{}'".format(opts.resume))
            checkpoint=torch.load(opts.resume)
            opts.start_epoch=checkpoint['epoch']
            best_val=checkpoint['best_val']
            SiameseNetModel.load_state_dict([checkpoint['state_dict']])
            SiameseNetOptim.load_state_dict([checkpoint['optimizer']])
        else:
            print("=> no checkpoint found at '{}'".format(opts.resume))
            best_val=float('inf')
    else:
        best_val=float('inf')

    
    ####################################################training#####################################################
    s_train=Siamese_Train(SiameseNetModel,Siamese_train_loader,criterion,optimizer)
    counter,loss_history=s_train.train()

    print ('Training Done.')
    func.show_plot(counter,loss_history)


    #dataiter=iter(test_loader)
    #x0,_,_ = next(dataiter)
    #test image
    #for i in range(10):
    #    _,x1,label2 = next(dataiter)
    #    concatenated = torch.cat((x0,x1),0)
    #    
    #    output1,output2 = net(Variable(x0).cuda(),Variable(x1).cuda())
    #    euclidean_distance = F.pairwise_distance(output1, output2)
    #    func.imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.cpu().data.numpy()[0][0]))

if __name__=="__main__":
    main()
