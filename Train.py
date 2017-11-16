#! /usr/bin/env python
#-*- coding:UTF-8 -*-

import torch
import torch.nn as nn
import torchvision 
import torch.utils.data 
import torch.optim as optim
from torch.autograd import Variable
from args import get_parser

#================================================
myparser=get_parser()
opts=myparser.parse_args()
#================================================

class Siamese_Train(object):
    def __init__(self,net,train_loader,criterion,optimizer):
        super(Siamese_Train,self).__init__()
        self.counter=[]
        self.loss_history=[]
        self.iteration_number=0
        self.train_loader=train_loader
        self.criterion=criterion
        self.net=net
        self.optimizer=optimizer

    def train(self):
        for epoch in range(0,opts.train_number_epochs):
            for i,data in enumerate(self.train_loader,0):
                img0,img1,label=data
                #print(type(img0),type(img1),type(label)
                img0,img1,label=Variable(img0).cuda(),Variable(img1).cuda(),Variable(label).cuda()
                output1,output2=self.net(img0,img1)
                self.optimizer.zero_grad()
                loss_contrastive=self.criterion(output1,output2,label)
                loss_contrastive.backward()
                self.optimizer.step()
                if i%10 ==0:
                    print('Epochs number {}\n Current loss:{}\n'.format(epoch,loss_contrastive.data[0]))
                    self.iteration_number+=10
                    self.counter.append(self.iteration_number)
                    self.loss_history.append(loss_contrastive.data[0])
        
        return self.counter,self.loss_history

