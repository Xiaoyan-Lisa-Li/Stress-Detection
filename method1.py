#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time 
import random
from torch.optim import Adam
from data_processing import *
from models import *
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models

### 
def plot_loss(losses, label):
    plt.figure(figsize=(10,5))
    plt.title("Training {}".format(label))
    plt.plot(losses,label=label)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
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
        
        
# #############################################################################
#          ### softmax        
#         test_acc += (preds.argmax(dim=1) == target).sum().float()
        


###############################################################################
### sigmoid        
        ### if preds > 0.5, preds = 1, otherwise, preds = 0
        # print(preds)
        preds = torch.tensor([p.item() > 0.5 for p in preds.cpu().detach().numpy()])

        test_acc += sum(torch.eq(y_test, preds))
       
        ## calculate how many samples are predicted correctly.
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
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        
    # optimizer = Adam(params=model.parameters(), lr=lr, amsgrad=False)
    criterion = nn.BCELoss().cuda()
     
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    criterion = nn.BCELoss().cuda()

    model.train()

    train_losses = []
    for epoch in range(num_epochs): 
        losses = 0
        avg_loss = 0.
        
        # print(train_loader)
        for i, sample_batch in enumerate(train_loader):
        
            x_train, y_train = sample_batch
            preds = model(x_train.cuda())
            optimizer.zero_grad()

            loss = criterion(preds.squeeze(), y_train.cuda().float())
            # loss = criterion(preds, y_train.cuda().long())
            
            loss.backward()
            optimizer.step()
            
            losses += loss.item()
            
        avg_loss = losses / len(train_loader)            
        train_losses.append(loss.item()) 
        
        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
       
        save_objs(model, epoch+1, avg_loss, optimizer, checkpoint_path)

    
    plot_loss(train_losses, label='train')

    
if __name__=="__main__":
    
    frame_size = (224,224)
    # frame_size = (28,28)
    num_epochs = 200
    batch_size = 32 
    lr = 1e-4
    
    SEED = 2019
    seed_everything(SEED)
    torch.cuda.is_available()
    
    checkpoint_path = './check_point_{}x{}/'.format(frame_size[0],frame_size[1])
    
    image_dir = './data/images_{}x{}/'.format(frame_size[0],frame_size[1])
    image_train_dir = './data/images_train_{}x{}/'.format(frame_size[0],frame_size[1])
    image_test_dir = './data/images_test_{}x{}/'.format(frame_size[0],frame_size[1])
    
    rest_csv = 'rest.csv'
    focus_csv = 'focus.csv'
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    ### data augmentation
    transform_t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    

    ## both trian and test data are from all 25 videos
    train_loader, test_loader, _ = create_datasets(batch_size,transform, transform_t, image_dir, rest_csv, focus_csv)
    
    # ### trian data is from first 21 videos and test data is from last 4 videos.
    # train_loader, test_loader = create_datasets2(batch_size,transform, transform_t, image_train_dir, image_test_dir, rest_csv, focus_csv)
    
    ##########################################################################
    ## using CNN with inputs size 28x28
    ##########################################################################
    model=CNN_Net().cuda()
    model.apply(initialize_weights)
    
    # ###########################################################################
    # ### or using pretrained vgg11 with inputs size 224x224
    # ###########################################################################
    # model = alexnet()
    
    ### train model
    # train_model(model, train_loader, num_epochs, checkpoint_path)
    
    predict(model, test_loader, checkpoint_path, 90)
    