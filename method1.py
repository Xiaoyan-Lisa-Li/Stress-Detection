#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fastprogress import master_bar, progress_bar
from torch.optim.lr_scheduler import CosineAnnealingLR
import time 
import random
from torch.optim import Adam
from data_processing import *
from models import CNN_Net

### 
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def initialize_weights(model):
  if isinstance(model, nn.Conv2d):
      nn.init.kaiming_uniform_(model.weight.data,nonlinearity='relu')
      if model.bias is not None:
          nn.init.constant_(model.bias.data, 0)
  elif isinstance(model, nn.BatchNorm2d):
      nn.init.constant_(model.weight.data, 1)
      nn.init.constant_(model.bias.data, 0)
  elif isinstance(model, nn.Linear):
      nn.init.kaiming_uniform_(model.weight.data)
      nn.init.constant_(model.bias.data, 0)

def train_model(model, train_loader, num_epochs):
    eta_min = 1e-5
    t_max = 10
    optimizer = Adam(params=model.parameters(), lr=lr, amsgrad=False)
    criterion = nn.MSELoss().cuda()
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)


    # best_epoch = -1
    # best_lwlrap = 0.
    mb = master_bar(range(num_epochs))
    
    for epoch in mb:
        # start_time = time.time()
        model.train()
        avg_loss = 0.

        for x_batch, y_batch in progress_bar(train_loader, parent=mb):
            preds = model(x_batch.cuda())
            loss = criterion(preds, y_batch.cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() / len(train_loader)


          
        scheduler.step()

        return

    
def predict_model():
    pass
    
if __name__=="__main__":
    
    num_epochs = 100
    batch_size = 32
    lr = 3e-3
    
    SEED = 2019
    seed_everything(SEED)
    torch.cuda.is_available()
    
    train_loader, test_loader = create_datasets()
   
    model=CNN_Net().cuda()
    model.apply(initialize_weights)
    
    train_model(model, train_loader, num_epochs)
    