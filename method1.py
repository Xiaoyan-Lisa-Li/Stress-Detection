#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from fastprogress import master_bar, progress_bar
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

def save_objs(model, epoch, loss, optimizer, save_path):
    path = save_path + 'model.{}'.format(epoch)   
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        "loss": loss,
    }, path)
    
def initialize_weights(model):
  if isinstance(model, nn.Conv2d):
      if model.weight is not None:
        nn.init.kaiming_uniform_(model.weight.data,nonlinearity='relu')
      if model.bias is not None:
          nn.init.constant_(model.bias.data, 0)
  elif isinstance(model, nn.BatchNorm2d):
      if model.weight is not None:
        nn.init.constant_(model.weight.data, 1)
      if model.bias is not None:  
        nn.init.constant_(model.bias.data, 0)
  elif isinstance(model, nn.Linear):
      if model.weight is not None:
        nn.init.kaiming_uniform_(model.weight.data)
      if model.bias is not None:
        nn.init.constant_(model.bias.data, 0)
        
def predict(model, test_loader, checkpoint_path, epoch):
    checkpoint = torch.load(checkpoint_path + 'model.{}'.format(epoch))
    model.load_state_dict(checkpoint["model_state_dict"]) 
        
    model.cuda()
    model.eval()
    test_acc = 0
    total_num = 0
    rest_true = 0
    rest_false = 0
    focus_true = 0
    focus_false = 0
    
    for i, sample_batch in enumerate(test_loader):
        x_test, y_test = sample_batch
        total_num += len(y_test)
        preds = model(x_test.cuda())
        print(preds)
        ### if preds > 0.5, preds = 1, otherwise, preds = 0
        preds = torch.tensor([p.item() > 0.5 for p in preds.cpu().detach().numpy()])
        test_acc += sum(torch.eq(y_test, preds))
        
        ### calculate how many samples are predicted correctly.
        for y, p in zip(y_test, preds):
            if y == p and y.item() == 0:
                rest_true += 1
            elif y != p and y.item() == 0:
                rest_false += 1
            elif y == p and y.item() == 1:
                focus_true += 1
            else:
                focus_false += 1
                
            
    print("rest_true is ",rest_true)   
    print("rest_false is ",rest_false)
    print("focus_true is ",focus_true)
    print("focus_false is ",focus_false)
    print("total number of samples is: ",total_num)

    print("test accuracy is {}".format(test_acc.item()/total_num))
    return test_acc/total_num

def train_model(model, train_loader, num_epochs, checkpoint_path):
    eta_min = 1e-5
    t_max = 10
        
    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        
    optimizer = Adam(params=model.parameters(), lr=lr, amsgrad=False)
    criterion = nn.MSELoss().cuda()
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)


    # best_epoch = -1
    # best_lwlrap = 0.
    # mb = master_bar(range(num_epochs))
    model.train()
    
    for epoch in range(num_epochs):    
        avg_loss = 0.
        print(train_loader)
        for i, sample_batch in enumerate(train_loader):
        
            x_train, y_train = sample_batch
            preds = model(x_train.cuda())
            optimizer.zero_grad()
            loss = criterion(preds, y_train.float().cuda())
            
            avg_loss += loss.item() / len(train_loader)
            
            loss.backward()
            optimizer.step()
            print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
                epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
       
        save_objs(model, epoch+1, avg_loss, optimizer, checkpoint_path)
        
        scheduler.step()

    
def predict_model():
    pass
    
if __name__=="__main__":
    
    num_epochs = 100
    batch_size = 32
    lr = 3e-3
    
    SEED = 2019
    seed_everything(SEED)
    torch.cuda.is_available()
    
    checkpoint_path = './check_point/'
    
    train_loader, test_loader = create_datasets(batch_size)
   
    model=CNN_Net().cuda()
    model.apply(initialize_weights)
    
    # train_model(model, train_loader, num_epochs, checkpoint_path)
    predict(model, test_loader, checkpoint_path, num_epochs)