#! /usr/bin/env python
#-*- coding:UTF-8 -*-
import time
import torch
import shutil
import numpy as np 
import torch.nn as nn
import torchvision 
import torch.utils.data 
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F 
from args import get_parser
from Deidentification import Scrambling #
from Loss import CenterLoss,RankingLoss
#================================================
myparser=get_parser()
opts=myparser.parse_args()
#================================================

class AverageMeter(object):
    """Compute and store the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0

    def update(self,val,n=1):
        self.val=val
        self.sum+=val*n
        self.count+=n
        self.avg=self.sum/self.count

def save_checkpoint(state,is_best,filename='checkpoint.path.tar'):
    """save file to checkpoint"""
    torch.save(state,filename)
    if is_best:
        shutil.copy(filename,'model_best.path.tar')

def Accuracy(output,target):
    """the evaluation index """
    res=0
    Min=np.argmin(output)
    Max=np.argmax(output)
    output_scaled=(output-output[Min])/(output[Max]-output[Min])
    res=abs(output_scaled-target)
    return res

class Siamese_Train(object):
    def __init__(self,model,train_loader,val_loader,criterion,optimizer):
        super(Siamese_Train,self).__init__()
        self.counter=[]
        self.loss_history=[]
        self.iteration_number=0
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.criterion=criterion
        self.model=model
        self.optimizer=optimizer

    def train_epoch(self,epoch):
        batch_time=AverageMeter()
        data_time=AverageMeter()
        top1=AverageMeter()
        top5=AverageMeter()
        end=time.time()

        #switch to train model
        self.model.train()

        for i,traindata in enumerate(self.train_loader,0):
            #measure data loading time
            data_time.update(time.time()-end)

            [img0,img1,label]=traindata
            opts.cuda=torch.cuda.is_available()
            if opts.cuda:
                img0,img1,label=Variable(img0).cuda(),Variable(img1).cuda(),Variable(label).cuda()
            output1,output2=self.model(img0,img1)
            self.optimizer.zero_grad()
            loss_contrastive=self.criterion(output1,output2,label)
            loss_contrastive.backward()
            self.optimizer.step()
            if i%10 ==0:
                print('Epochs number {}\n Current loss:{}\n'.format(epoch,loss_contrastive.data[0]))
                self.iteration_number+=10
                self.counter.append(self.iteration_number)
                self.loss_history.append(loss_contrastive.data[0])

    def validate(self,val_loader,criterion):
        batch_time=AverageMeter()
        top1=AverageMeter()
        val_loss=AverageMeter()
        
        #switch to evaluate mode
        self.model.eval()
        end=time.time()
        for i,valdata in enumerate(val_loader,0):
            batch_time.update(i)
            input_var=list()
            [img0,img1,label]=valdata
            opts.cuda=torch.cuda.is_available()
            img0,img1,label=Variable(img0,volatile=True),Variable(img1,volatile=True),Variable(label,volatile=True)
            if opts.cuda:
                img0,img1,label=img0.cuda(),img1.cuda(),label.cuda()
            output1,output2=self.model(img0,img1)
            #use euclidean distance of two image
            euclidean_distance =F.pairwise_distance(output1,output2)#
                       
            output=euclidean_distance.cpu().data.numpy()
            target=label.cpu().data.numpy()
            loss_=self.criterion(output1,output2,label)
            val_loss.update(loss_)
            res=Accuracy(output,target)
            top1.update(res)
        print('validate :abs(output_euclidean_distance-label) => {} '.format(top1.avg[0]))

        return top1.avg[0]

    def train(self,best_val):
        valtrack = 0
        for epoch in range(opts.Siamese_Start_epoch,opts.Siamese_train_number_epochs):#from checkpoint on
            
            #train for one epoch
            self.train_epoch(epoch)

            if (epoch+1)%opts.valfre ==0 and epoch !=0:#save the model
                val_loss=self.validate(self.val_loader,self.criterion)

                #save the best model
                is_best=val_loss<best_val
                best_val=min(val_loss,best_val)
                save_checkpoint({
                    'epoch':epoch+1,
                    'state_dict':self.model.state_dict(),
                    'best_val':best_val,
                    'optimizer':self.optimizer.state_dict(),
                    'curr_val':val_loss,
                },is_best)

                print('** Validation : %f (best) '%(best_val))

        return self.counter,self.loss_history

class Reidentify_Train(object):
    def __init__(self,Orimodel,Deidenmodel,train_loader,val_loader,optimizer):
        super(Deidentify_Train,self).__init__()
        self.Orimodel=Orimodel
        self.Deidenmodel=Deidenmodel
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.optimizer=optimizer

    def train(self,best_val):
        for epoch in range(opts.Deidentify_Start_epoch,opts.Deidentify_train_number_epochs):
            #train for one epoch
            self.train_epoch(epoch,self.Orimodel,self.Deidenmodel,self.train_loader)

            '''if (epoch+1)%opts.valfre ==0 and epoch !=0:#save the model
                val_loss=self.validate(self.val_loader,self.Orimodel,self.Deidenmodel)
                #save the best model
                is_best=val_loss<best_val
                best_val=min(val_loss,best_val)
                save_checkpoint({
                    'epoch':epoch+1,
                    'state_dict':self.model.state_dict(),
                    'best_val':best_val,
                    'optimizer':self.optimizer.state_dict(),
                    'curr_val':val_loss,
                },is_best)

                print('** Validation : %f (best) '%(best_val))

        return self.counter,self.loss_history'''
            
    def train_epoch(self,epoch,Orimodel,Deidenmodel,train_loader):
        batch_time=AverageMeter()
        data_time=AverageMeter()
        top1=AverageMeter()
        end=time.time()

        #switch to train model
        model.train()

        for i,traindata in enumerate(train_loader,0):
            #measure data loading time
            data_time.update(time.time()-end)

            [img,label]=traindata#load original training data
            img_scrambled=Scrambling(img,opts.n,opts.a,opts.b)

            opts.cuda=torch.cuda.is_available()
            if opts.cuda:
                img, img_scrambled,label=Variable(img).cuda(),Variable(img_scrambled).cuda(),Variable(label).cuda()
            
            #load data into models
            output1=Orimodel(img)
            center_loss,Orimodel._buffers['centers']=CenterLoss(Orimodel._buffers['centers'],Orimodel.features,label,opts.alpha,opts.bit_length)
            output2=Deidenmodel(img_scrambled)                 
            ranking_loss=RankingLoss(output1,output2,label)
            loss_=ranking_loss+center_loss#connect loss
            self.optimizer.zero_grad()
            loss_.backward()
            self.optimizer.step()
            if i%10 ==0:
                print('Epochs number {}\n Current loss:{}\n'.format(epoch,loss_.data[0]))
                self.iteration_number+=10
                self.counter.append(self.iteration_number)
                self.loss_history.append(loss_.data[0])

    def validate(self,Orimodel,Deidenmodel):
        batch_time=AverageMeter()
        top1=AverageMeter()
        val_loss=AverageMeter()
        
        #switch to evaluate mode
        self.model.eval()
        end=time.time()
        for i,valdata in enumerate(val_loader,0):
            batch_time.update(i)
            input_var=list()
            [img,label]=valdata
            img_scrambled=Scrambling(img,opts.n,opts.a,opts.b)
            opts.cuda=torch.cuda.is_available()
            img,img_scrambled,label=Variable(img,volatile=True),Variable(img_scrambled,volatile=True),Variable(label,volatile=True)
            if opts.cuda:
                img,img_scrambled,label=img.cuda(),img_scrambled.cuda(),label.cuda()

            output1=Orimodel(img)
            output2=Deidenmodel(img_scrambled)  
            #use euclidean distance of two image
            euclidean_distance =F.pairwise_distance(output1,output2)#
                       
            output=euclidean_distance.cpu().data.numpy()
            target=label.cpu().data.numpy()
            loss_=self.criterion(output1,output2,label)
            val_loss.update(loss_)
            res=Accuracy(output,target)
            top1.update(res)
        print('validate :abs(output_euclidean_distance-label) => {} '.format(top1.avg[0]))

        return top1.avg[0]
