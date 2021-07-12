#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import facial_data_process
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CNN_2d(nn.Module):
    def __init__(self):
        super(CNN_2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
                                kernel_size=3,
                                stride=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv1_drop = nn.Dropout2d(0.1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv2_drop = nn.Dropout2d(0.1)
        self.dense1 = nn.Linear(in_features=1600, out_features=128)
        self.dense2 = nn.Linear(128, 1)

    def forward(self, input):
        # print("input.shape",input.size())
        x = self.conv1_drop(F.max_pool2d(self.conv1_bn(F.relu(self.conv1(input))), 2))       
        x = self.conv2_drop(F.max_pool2d(self.conv2_bn(F.relu(self.conv2(x))), 2))
        x = x.view(-1, 1600) #reshape
        x = F.relu(self.dense1(x))
        x = F.sigmoid(self.dense2(x))
        return x
# net = CNN_Net()
# print(net)

class CNN_1d(nn.Module):
    def __init__(self):
        super(CNN_1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64,
                                kernel_size=3,
                                stride=1)
        self.conv1_bn = nn.BatchNorm1d(64)
        self.maxpool1 = nn.MaxPool1d(3)
        self.conv1_drop = nn.Dropout(0.1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3)
        self.conv2_bn = nn.BatchNorm1d(32)
        self.maxpool2 = nn.MaxPool1d(3)
        self.conv2_drop = nn.Dropout(0.1)      
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3)
        self.conv3_bn = nn.BatchNorm1d(32)
        self.maxpool3 = nn.MaxPool1d(3)
        self.conv3_drop = nn.Dropout(0.1)        
        self.dense1 = nn.Linear(in_features=320, out_features=32)
        self.dense2 = nn.Linear(32, 1)

    def forward(self, input):
        # print("input.shape",input.size())
        input = input.unsqueeze(1)
        x = self.conv1_drop(self.maxpool1(self.conv1_bn(F.relu(self.conv1(input)))))       
        x = self.conv2_drop(self.maxpool2(self.conv2_bn(F.relu(self.conv2(x)))))
        x = self.conv3_drop(self.maxpool3(self.conv3_bn(F.relu(self.conv3(x)))))
        x = x.view(-1, 320) #reshape
        x = F.relu(self.dense1(x))
        x = F.sigmoid(self.dense2(x))
        return x

net = CNN_1d()
print(net)

        
class alexnet(nn.Module):

    def __init__(self, num_classes: int = 1) -> None:
        super(alexnet, self).__init__()
        self.net = models.vgg11(pretrained=True)
        for p in self.net.features.parameters():
            p.requires_grad=False
        self.net.classifier[-1] = nn.Linear(4096, num_classes)
        self.net.cuda()
        
    def forward(self,x):
        x = self.net(x)
        x = F.sigmoid(x)
        return x
        
# alex_net = alexnet()
# print(alex_net)
    

