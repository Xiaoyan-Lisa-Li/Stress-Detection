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
from facial_data_process import *
from models import *
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
from SVM import svm_f, svm_ecg
from utils import *
from datetime import datetime
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset
from ecg_data_process import create_ecg_data

   
def predict(model, test_loader, checkpoint_path, epoch, method, results_path, fold, n):
    
    results_f = results_path + '{}_restults.txt'.format(method)
    
    # checkpoint = torch.load(checkpoint_path + 'model{}_repeat{}_fold{}.pth'.format(epoch, n ,fold))
    # model.load_state_dict(checkpoint["model_state_dict"]) 
        
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
        pred = pred.squeeze()
        ### if preds > 0.5, preds = 1, otherwise, preds = 0       
        pred = [p.item() > 0.5 for p in pred.cpu().detach().numpy()]

        pred = list(map(int, pred))
        
        print('true y is',y)
        print("predicted y is",pred)

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
    rest_precision = rest_true/(rest_true+focus_false)
    focus_precision = focus_true/(focus_true+rest_false)
    print("This is the {}th repeat {}th fold".format(n, fold))            
    print("rest_true is ",rest_true)   
    print("rest_false is ",rest_false)
    print("focus_true is ",focus_true)
    print("focus_false is ",focus_false)
    print('rest precision is',rest_precision)
    print('focus precision is',focus_precision)
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
        f.writelines("the rest precision is {} \n".format(rest_precision))
        f.writelines("the focus precision is {} \n".format(focus_precision))
        f.writelines("The test accracy of {} is {} \n".format(method, acc))

   
    fig_path = results_path + "/1d_cnn_figures/"
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
       
    plot_confusion_matirx(y_true, y_pred, method, fig_path, fold, n, labels = [0,1])
    
    return acc,  rest_precision, focus_precision


def train_model(model, train_loader, num_epochs, checkpoint, results_path,fold, n):
    
    checkpoint_path = results_path + checkpoint
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        
    # optimizer = Adam(params=model.parameters(), lr=lr, amsgrad=False)
     
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
    
    checkpoint = 'check_point/'

    k_folds = 5
    repeat = 3
    time_s = 540
      # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    x,y = create_ecg_data(time_s)
    
    tensor_x = torch.Tensor(x) 
    tensor_y = torch.Tensor(y)
    dataset = TensorDataset(tensor_x,tensor_y)
    test_acc = []
    rest_prec_ls = []
    focus_prec_ls = []
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

            model=CNN_1d().cuda()
            model.apply(reset_weights)

            checkpoint_path = args.results + checkpoint
            
            # train model
            train_model(model, train_loader, args.num_epochs, checkpoint, args.results, fold, n)

            acc, rest_precision, focus_precision= predict(model, test_loader, checkpoint_path, args.num_epochs-1, args.method, args.results, fold, n)
            test_acc.append(acc)

            focus_prec_ls.append(focus_precision)
            rest_prec_ls.append(rest_precision)
            
    acc_mean = np.array(test_acc).mean()
    acc_std = np.array(test_acc).std()

    rest_mean = np.array(rest_prec_ls).mean()
    rest_std = np.array(rest_prec_ls).std()
    
    focus_mean = np.array(focus_prec_ls).mean()
    focus_std = np.array(focus_prec_ls).std()
    
    print("Method %s: %0.4f accuracy with a standard deviation of %0.4f" % (args.method, acc_mean, acc_std))
    print("Method %s: %0.4f rest precision with a standard deviation of %0.4f" % (args.method, rest_mean, rest_std))
    print("Method %s: %0.4f focus precision with a standard deviation of %0.4f" % (args.method, focus_mean, focus_std))
    
    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    results_f = args.results + '{}_restults.txt'.format(args.method)
    with open(results_f, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        f.write("Method %s: %0.2f accuracy with a standard deviation of %0.2f \n" % (args.method, acc_mean, acc_std))      
        f.write("Method %s: %0.4f rest precision with a standard deviation of %0.4f \n" % (args.method, rest_mean, rest_std))
        f.write("Method %s: %0.4f focus precision with a standard deviation of %0.4f \n" % (args.method, focus_mean, focus_std))

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_epochs', type=int, default=2000,
                        help='')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='')
    parser.add_argument('--method', type=str, default='1d cnn',
                        help='')    
    parser.add_argument('--results', type=str, default='./ecg_results/',
                        help='')  
    
    args = parser.parse_args()
    
    # seed_everything(args.seed)
    
    torch.cuda.is_available()
    
    if not os.path.exists(args.results):
        os.makedirs(args.results)
    
    
    if args.method == "1d cnn":
        main(args)

    if args.method == "svm":
        svm_ecg(args.results, args.method)


    
