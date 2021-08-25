#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import facial_data_process
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils import weight_norm

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
      if hasattr(layer, 'reset_parameters'):
          print(f'Reset trainable parameters of layer = {layer}')
          layer.reset_parameters()
    
def reset_weights_vgg(alex_net):    
    for layer in alex_net.net.classifier:
        if type(layer) == nn.Linear:
            layer.reset_parameters()
            print(f'Reset trainable parameters of layer = {layer}')
            
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
        #                                    stride=stride, padding=padding, dilation=dilation))
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.conv1_bn = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)

        # self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
        #                                    stride=stride, padding=padding, dilation=dilation))
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.conv2_bn = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.conv1_bn, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.conv2_bn, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return F.log_softmax(o, dim=1)
    
    

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
        x0 = x.view(-1, 1600) #reshape
        x1 = F.relu(self.dense1(x0))
        x = F.sigmoid(self.dense2(x1))
        return x,x0
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
        self.dense1 = nn.Linear(in_features=320, out_features=32) ### 3seconds in_features=320; 6seconds in_features=672;
        self.dense2 = nn.Linear(32, 1)

    def forward(self, input):
        # print("input.shape",input.size())
        input = input.unsqueeze(1)
        x = self.conv1_drop(self.maxpool1(self.conv1_bn(F.relu(self.conv1(input)))))       
        x = self.conv2_drop(self.maxpool2(self.conv2_bn(F.relu(self.conv2(x)))))
        x = self.conv3_drop(self.maxpool3(self.conv3_bn(F.relu(self.conv3(x)))))
        x_0 = x.view(-1, 320) ###reshape 3 seconds (-1,320); 6seconds view(-1, 672);
        x_1 = F.relu(self.dense1(x_0))
        x = torch.sigmoid(self.dense2(x_1))
        return x, x_0

# net = CNN_1d()
# print(net)

        
class alexnet(nn.Module):

    def __init__(self, num_classes: int = 1) -> None:
        super(alexnet, self).__init__()
        self.net = models.vgg11(pretrained=True)
        self.features = nn.Sequential(*list(self.net.children())[:-1])
        for p in self.net.features.parameters():
            p.requires_grad=False
        for p in self.net.avgpool.parameters():
            p.requires_grad=False
        # for p in self.net.classifier[:-5].parameters():
        #     p.requires_grad=False        
        self.net.classifier[-1] = nn.Linear(4096, num_classes)
        
       
    def forward(self,input):
        x0 = self.net(input)
        x = torch.sigmoid(x0)
        x_f = self.features(input)
        x_f = x_f.view(-1,25088)
        return x, x_f
        
alex_net = alexnet()
print(alex_net)
pytorch_total_params = sum(p.numel() for p in alex_net.parameters())
print(pytorch_total_params)
pytorch_total_params = sum(p.numel() for p in alex_net.parameters() if p.requires_grad)
print(pytorch_total_params)

    

