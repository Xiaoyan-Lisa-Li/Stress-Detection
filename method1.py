#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
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
from SVM import svm_f
from utils import *
from datetime import datetime
from sklearn.model_selection import KFold

   
def predict(model, test_loader, checkpoint_path, epoch, method, results_path, fold, n):
    
    results_f = results_path + '{}_restults.txt'.format(method)
    
    checkpoint = torch.load(checkpoint_path + 'model{}_repeat{}_fold{}.pth'.format(epoch, n ,fold))
    model.load_state_dict(checkpoint["model_state_dict"]) 
        
    model.cuda()
    model.eval()
    test_acc = 0
    total_num = 0
    rest_true = 0
    rest_false = 0
    focus_true = 0
    focus_false = 0
    y_pred = []
    y_true = []

    for i, sample_batch in enumerate(test_loader):
        x_test, y = sample_batch
        total_num += len(y)
        pred = model(x_test.cuda())
  
        ### if preds > 0.5, preds = 1, otherwise, preds = 0
        print(y)
        pred = [p.item() > 0.5 for p in pred.cpu().detach().numpy()]

        pred = list(map(int, pred))

        y_pred.extend(pred)
        
        y = y.detach().numpy()
        
        y_true.extend(y.tolist())
        
        test_acc += sum(y == np.array(pred))
        
       
        ## calculate how many samples are predicted correctly.
        for t, p in zip(y, pred):
            if t == p and t.item() == 0:
                rest_true += 1
            elif t != p and t.item() == 0:
                rest_false += 1
            elif t == p and t.item() == 1:
                focus_true += 1
            else:
                focus_false += 1
    print("This is the {}th repeat {}th fold".format(n, fold))            
    print("rest_true is ",rest_true)   
    print("rest_false is ",rest_false)
    print("focus_true is ",focus_true)
    print("focus_false is ",focus_false)
    print("total number of samples is: ",total_num)
    acc = test_acc.item()/total_num
    print("test accuracy is {}".format(acc))
    
    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    
    with open(results_f, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        f.writelines("repeat {}, fold {} \n".format(n, fold))
        f.writelines("the number of rest samples that are correctly classified is {} \n".format(rest_true))
        f.writelines("the number of rest samples that are incorrectly classified is {} \n".format(rest_false))
        f.writelines("the number of focus samples that are correctly classified is {} \n".format(focus_true))
        f.writelines("the number of focus samples that are incorrectly classified is {} \n".format(focus_false))
        f.writelines("The test accracy of {} is {} \n".format(method, acc))
    
    # print(y_true)
    # print(y_pred)
        if method == '2d cnn':
            fig_path = results_path + "/2d_cnn_figures/"
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
        elif method == 'pretrained vgg':
            fig_path = results_path + "/pretain_vgg_figures/"
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
    plot_confusion_matirx(y_true, y_pred, method, fig_path, fold, n, labels = [0,1])
    
    return acc

def train_model(model, train_loader, num_epochs, checkpoint, results_path,fold, n):
    
    checkpoint_path = results_path + checkpoint
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
        
        print('Repeat {} Fold {} Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(n, fold,
            epoch, num_epochs, i, len(train_loader), loss.item()))
        
        
        if (epoch+1) %10 == 0:
            save_path = checkpoint_path + 'model{}_repeat{}_fold{}.pth'.format(epoch,n,fold)  
            save_objs(model, epoch, avg_loss, optimizer, save_path)

    
    plot_loss(args.method, train_losses, results_path, label='train')
    
    

def main(args):
    
    checkpoint = 'check_point_{}x{}/'.format(args.frame_size[0],args.frame_size[1])
    
    image_dir = './data/images_{}x{}/'.format(args.frame_size[0],args.frame_size[1])
    image_train_dir = './data/images_train_{}x{}/'.format(args.frame_size[0],args.frame_size[1])
    image_test_dir = './data/images_test_{}x{}/'.format(args.frame_size[0],args.frame_size[1])
    
    rest_csv = 'rest.csv'
    focus_csv = 'focus.csv'
    
    k_folds = 5
    repeat = 3
    
      # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

   
    ## both trian and test data are from all 25 videos
    train_loader, test_loader, dataset = create_datasets(args.batch_size,transform, image_dir, rest_csv, focus_csv)
    
    # ### trian data is from first 21 videos and test data is from last 4 videos.
    # train_loader, test_loader = create_datasets2(args.batch_size,transform, image_train_dir, image_test_dir, rest_csv, focus_csv)
    
    test_acc = []
    for n in range(repeat):
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            
            # Print
            print(f'FOLD {fold}')
            print('--------------------------------')  
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            
            # Define data loaders for training and testing data in this fold
            train_loader = torch.utils.data.DataLoader(
                              dataset, batch_size=args.batch_size, sampler=train_subsampler)
            test_loader = torch.utils.data.DataLoader(
                              dataset, batch_size=args.batch_size, sampler=test_subsampler)
            if args.method == '2d cnn':
                model=CNN_Net().cuda()
                model.apply(reset_weights)
                # model.apply(initialize_weights)
            elif args.method == 'pretrained vgg':
                model = alexnet()
                model.apply(reset_weights_vgg)
                
            checkpoint_path = args.results + checkpoint
            
            ## train model
            train_model(model, train_loader, args.num_epochs, checkpoint, args.results, fold, n)

            acc = predict(model, test_loader, checkpoint_path, args.num_epochs-1, args.method, args.results, fold, n)
            test_acc.append(acc)
            
    mean = np.array(test_acc).mean()
    std = np.array(test_acc).std()
    print("Method %s: %0.2f accuracy with a standard deviation of %0.2f" % (args.method, mean, std))
    
    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    results_f = args.results + '{}_restults.txt'.format(args.method)
    with open(results_f, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        f.write("Method %s: %0.2f accuracy with a standard deviation of %0.2f \n" % (args.method, mean, std))    

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='')
    parser.add_argument('--frame_size', type=tuple, default=(28,28),
                        help='')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='')
    parser.add_argument('--method', type=str, default='pretrained vgg',
                        help='')    
    parser.add_argument('--seed', type=int, default=2021,
                        help='')    
    parser.add_argument('--results', type=str, default='./method1_results/',
                        help='')  
    
    args = parser.parse_args()
    
    # seed_everything(args.seed)
    
    torch.cuda.is_available()
    
    if not os.path.exists(args.results):
        os.makedirs(args.results)
    
    
    if args.method == "2d cnn":
        args.frame_size = (28,28)
        main(args)
    if args.method == "pretrained vgg":
        args.frame_size = (224,224)
        main(args)
    if args.method == "svm":
        args.frame_size = (28,28)
        svm_f(args.batch_size, args.frame_size,args.results, args.method)


    