#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import data_processing
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
                               kernel_size=3,
                               stride=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv1_drop = nn.Dropout2d(0.3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv2_drop = nn.Dropout2d(0.3)
        self.dense1 = nn.Linear(in_features=1600, out_features=128)
        self.dense2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1_drop(F.max_pool2d(self.conv1_bn(F.relu(self.conv1(x)), 2)))       
        x = self.conv2_drop(F.max_pool2d(self.conv2_bn(F.relu(self.conv2(x)), 2)))
        x = x.view(-1, 1600) #reshape
        x = F.relu(self.dense1_bn(self.dense1(x)))
        x = F.sigmoid(self.dense2(x))
        return x


net = CNN_Net()
print(net)