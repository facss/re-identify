#! /usr/bin/env python
#-*- coding:UTF-8 -*-

import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import os
import random
import numpy as np

#import orl face dataset
class ORLFaceData(Dataset):
    def __init__(self,root_dir,transform=None,should_invert=True):
        super(ORLFaceData,self).__init__()
        self.root_dir=root_dir
        self.imglist=self.loadimglist()
        self.should_invert=should_invert
        self.transform=transform
    
    def loadimglist(self):
        #return image name list
        img_list=list()
        tmp_imglist=list()
        for tr ,dirs,files in os.walk(self.root_dir):
            for f in files:
                img_name=os.path.join(self.root_dir,f)
                tmp_imglist.append(img_name)
        sorted_list=sorted(tmp_imglist)#sort the image name list
        #print(len(sorted_list))           

        if 'training' in self.root_dir:
            for i in range(1,41):
                for j in range(0,10):
                    img_list.append((sorted_list[(i-1)*10+j],i))#tuple(image,class) 
        if 'testing' in self.root_dir:
            for i in range(1,41):
                for j in range(0,10):
                    img_list.append((sorted_list[(i-1)*10+j],i))#tuple(image,class)
        return img_list  
    
    def __getitem__(self,item):
        if 'training' in self.root_dir:
            img0_tuple=random.choice(self.imglist)
            #we need to make sure approx 50% of images are in the same class
            should_get_same_class=random.randint(0,1)# 50% 0 or 1

            if should_get_same_class:
                while True:
                    #keep looping till the same class image is found
                    img1_tuple=random.choice(self.imglist)
                    if img0_tuple[1]==img1_tuple[1]:
                        break
            else:
                while True:
                    #keep looping till the different class image is found
                    img1_tuple=random.choice(self.imglist)
                    if img0_tuple[1] !=img1_tuple[1]:
                        break

            img0=Image.open(img0_tuple[0]).convert('L')
            img1=Image.open(img1_tuple[0]).convert('L')
        
            if  self.should_invert:
                img0=PIL.ImageOps.invert(img0)
                img1=PIL.imageOps.invert(img1)

            if self.transform is not None:
                img0=self.transform(img0)
                img1=self.transform(img1)
        
            labl=torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
            return img0,img1,labl#pair images and whether they are the same
        else:
            img=Image.open(self.imglist[item][0]).convert('L')
            label=self.imglist[item][1]
            if self.transform is not None:
                img=self.transform(img)
            return img,label
        
    def __len__(self):
        return len(self.imglist)

