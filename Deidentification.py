#! /usr/bin/env python
#-*- coding:UTF-8 -*-
import torch
import torch.nn as nn 
import numpy as np 
import PIL 

class Scrambling(object):
    def __init__(self,img,n,a,b):
        #input image:a 
        super(Scrambling,self).__init__()
        self.n=n 
        img=img.cpu.data.numpy()
        h=img.shape[0]
        w=img.shape[1]
        imgn=[[0]*h]*w
        for i in range(n ):
            for y in range(h):
                for x in range(w):
                    xx=((x-1)+b*(y-1) )%h+1
                    yy=(a*(x-1)+(a*b+1)*(y-1))%h+1
                    imgn[yy,xx]=img[y,x]
            img=imgn
        
        return imgn