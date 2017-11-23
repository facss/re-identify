#! /usr/bin/env python
#-*- coding:UTF-8 -*-

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import BasicBlock,ResNet,model_urls
import math

#siamese network
class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet,self).__init__()
        self.cnn1=nn.Sequential(
            nn.ReflectionPad2d(1),#表示对边缘进行pad增加
            nn.Conv2d(1,4,kernel_size=3),#conv层
            nn.ReLU(inplace=True),#表示新创建一个对象并直接对这个对象进行修改？？
            nn.BatchNorm2d(4),# With Learnable Parameters;num_features: num_features from an expected input of size batch_size x num_features x height x width
            nn.Dropout2d(p=.2),#p:probability of an element to be zeroed

            nn.ReflectionPad2d(1),
            nn.Conv2d(4,8,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8,8,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),
        )
        self.fc1=nn.Sequential(
            nn.Linear(8*100*100,500),
            nn.ReLU(inplace=True),
            
            nn.Linear(500,500),
            nn.ReLU(inplace=True),

            nn.Linear(500,5)
        )
    def forward_once(self,x):
        output=self.cnn1(x)
        output=output.view(output.size()[0],-1)
        output=self.fc1(output)
        return output

    def forward(self,input1,input2):
        output1=self.forward_once(input1)
        output2=self.forward_once(input2)
        return output1,output2

#original face cnn network 
class OriNet(nn.Module):
    def __init__(self,bits_len,pretrained=False,**kwargs):
        super(OriNet,self).__init__()
        self.bits_len=bits_len
        self.resnetmodel=ResNet(BasicBlock,[2,2,2,2],**kwargs)
        if pretrained:
            parameters=model_zoo.load_url(model_urls['resnet18'])
            self.model.load_state_dict(parameters)
        self.resnetmodel.avgpool=None
        self.resnetmodel.fc1=nn.Linear(512*3*4,512)
        self.resnetmodel.fc2=nn.Linear(512,512)
        self.resnetmodel.classifier=nn.Linear(512,bits_len)
        self.register_buffer('centers',torch.zeros(bits_len,512))

    def forward(self,input1):
        input1=self.resnetmodel.conv1(input1)
        input1=self.resnetmodel.bn1(input1)
        input1=self.resnetmodel.relu(input1)
        input1=self.resnetmodel.maxpool(input1)

        input1=self.resnetmodel.layer1(input1)
        input1=self.resnetmodel.layer2(input1)
        input1=self.resnetmodel.layer3(input1)
        input1=self.resnetmodel.layer4(input1)

        input1=input1.view(input1.size(),-1)
        input1=self.resnetmodel.fc1(input1)
        #feature for center loss
        input1=self.resnetmodel.fc2(input1)
        self.features=input1
        input1=self.resnetmodel.classifier(input1)
        output1=F.log_softmax(input1)

        return output1

# de-identification face model
class DeidenNet(nn.Module):
    def __init__(self,bits_len,pretrained=False,**kwargs):
        super(DeidenNet,self).__init__()
        self.deidentifymodel.fc1=nn.Linear(8*100*100,500)
        self.deidentifymodel.fc2=nn.Linear(500,bits_len)
    
    def forward(self,input2):         
        #full connect network
        input2=self.deidentifymodel.fc1(input2).clamp(min=0)
        input2=F.relu(input2)
        input2=self.deidentifymodel.fc2(input2)
        input2=F.relu(input2)
        output2=F.log_softmax(input2)

        return output2

