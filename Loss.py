#! /usr/bin/env python
#-*- coding:UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self,margin=2.0):
        super(ContrastiveLoss,self).__init__()
        self.margin=margin

    def forward(self,output1,output2,label):
        euclidean_distance=F.pairwise_distance(output1,output2)
        loss_contrastive=torch.mean((1-label)*torch.pow(euclidean_distance,2)+label*torch.pow(torch.clamp(self.margin-euclidean_distance,min=0.0),2))

        return loss_contrastive

class CenterLoss(nn.Module):
    def __init__(self,centers,features,target,alpha,bit_len):
        super(CenterLoss,self).__init__()
        batch_size=target.size(0)
        features_dim=features.size(1)

        target_expand=target.view(batch_size,1).expand(batch_size,features_dim)
        centers_var=Variable(centers)
        centers_batch=centers_var.gather(0,target_expand)
        criterion=nn.MSELoss()
        center_loss=criterion(features,center_batch)

        diff=center_batch-features
        unique_label,unique_reverse,unique_count=np.unique(target.cpu().data.numpy(),return_inverse=True,return_counts=True)
        appear_times=torch.from_numpy(unique_count).gather(0,torch.from_numpy(unique_reverse))
        appear_times_expand=appear_times.view(-1,1).expand(batch_size,features_dim).type(torch.FloatTensor)
        diff_cpu=diff.cpu().data/appear_times_expand.add(1e-6)
        diff_cpu=alpha*diff_cpu
        for i in range(batch_size):
            centers[target.data[i]]-=diff_cpu[i].type(centers.type())

        return center_loss,centers

class RankingLoss(nn.Module):
    #two tensor
    def __init__(self,output1,output2,label):
        super(RankingLoss,self).__init__()
        print(output1)
        print('---------------------------')
        print(output2)
        return label