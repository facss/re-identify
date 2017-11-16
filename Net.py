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
            nn.BatchNorm2d(4),# With Learnable Parameters;num_features: num_features from an expected input of size batch_size x num_features x height x width作用？？
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
        self.model=ResNet(BasicBlock,[2,2,2,2],**kwargs)
        if pretrained:
            parameters=model_zoo.load_url(model_urls['resnet18'])
            self.model.load_state_dict(parameters)
        self.model.avgpool=None
        self.model.fc1=nn.Linear(512*3*4,512)
        self.model.fc2=nn.Linear(512,512)
        self.model.classifier=nn.Linear(512,num_classes)
        self.register_buffer('ranking',torch.zeros(num_classes,512))
        self.bits_len=bits_len

    def forward(self,x):
        x=self.model.conv1(x)
        x=self.model.bn1(x)
        x=self.model.relu(x)
        x=self.model.maxpool(x)

        x=self.model.layer1(x)
        x=self.model.layer2(x)
        x=self.model.layer3(x)
        x=self.model.layer4(x)

        x=x.view(x.size(),-1)
        x=self.model.fc1(x)
        #feature for center loss
        x=self.model.fc2(x)
        self.features=x
        x=self.model.classifier(x)
        return F.log_softmax(x)

#de-identification face model
#class DeidenNet(nn.Module):
#    def __init__(self):
#        super(DeidenNet,self).__init__()
#        nn.Linear()

    
#    def forward(self):


