#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Train the model using only 24 videos, and 1 video for testing.
'''

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
from SVM import svm_f
from utils import *
from datetime import datetime

def plot_confusion_matirx(y_true, y_pred, method, fig_path, labels):
    np.set_printoptions(precision=4)
    
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['rest','focus'])

    
    # NOTE: Fill all variables here with default values of the plot_confusion_matrix
    disp = disp.plot(include_values=True,
                     cmap=plt.cm.Blues, ax=None, xticks_rotation='horizontal')
    

    plt.savefig(fig_path + "{}_confusion_matrix.png".format(method))
    plt.show()
   
def predict(model, test_loader, checkpoint_path, epoch, method, results_path):
    
    results_f = results_path + '{}_restults.txt'.format(method)
    
    checkpoint = torch.load(checkpoint_path + 'model{}.pth'.format(epoch))
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
        # print(preds)
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
    rest_precision = rest_true/(rest_true+focus_false)
    focus_precision = focus_true/(focus_true+rest_false)        
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
        f.writelines("the number of rest samples that are correctly classified is {} \n".format(rest_true))
        f.writelines("the number of rest samples that are incorrectly classified is {} \n".format(rest_false))
        f.writelines("the number of focus samples that are correctly classified is {} \n".format(focus_true))
        f.writelines("the number of focus samples that are incorrectly classified is {} \n".format(focus_false))
        f.writelines("the rest precision is {} \n".format(rest_precision))
        f.writelines("the focus precision is {} \n".format(focus_precision))
        f.writelines("The test accracy of {} is {} \n".format(method, acc))    

    plot_confusion_matirx(y_true, y_pred, method, results_path, labels = [0,1])

def train_model(model, train_loader, num_epochs, checkpoint, results_path):
    
    checkpoint_path = results_path + checkpoint
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        
     
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
            epoch, num_epochs, i, len(train_loader), loss.item()))
        
        checkpoint_f = checkpoint_path + 'model{}.pth'.format(epoch)  
        save_objs(model, epoch, avg_loss, optimizer, checkpoint_f)

    
    plot_loss(args.method, train_losses, results_path, label='train')
    
    

def main(args, model):
    
    checkpoint = 'check_point_{}x{}/'.format(args.frame_size[0],args.frame_size[1])
    
    image_dir = './data/images_{}x{}/'.format(args.frame_size[0],args.frame_size[1])
    image_train_dir = './data/data2/images_train_{}x{}/'.format(args.frame_size[0],args.frame_size[1])
    image_test_dir = './data/data2/images_test_{}x{}/'.format(args.frame_size[0],args.frame_size[1])
    
    rest_csv = 'rest.csv'
    focus_csv = 'focus.csv'
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
     
    ### trian data is from first 24 videos and test data is from last 1 videos.
    train_loader, test_loader = create_datasets2(args.batch_size,transform, image_train_dir, image_test_dir, rest_csv, focus_csv)
    
    ## train model
    # train_model(model, train_loader, args.num_epochs, checkpoint, args.results)
    
    checkpoint_path = args.results + checkpoint
    predict(model, test_loader, checkpoint_path, args.num_epochs-1, args.method, args.results)
    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='')
    parser.add_argument('--frame_size', type=tuple, default=(28,28),
                        help='')
    parser.add_argument('--num_epochs', type=int, default=100, ### 40 for pretrained vgg; 100 for 2D CNN
                        help='')
    parser.add_argument('--method', type=str, default='2d cnn',
                        help='')    
    parser.add_argument('--seed', type=int, default=2021,
                        help='')    
    parser.add_argument('--results', type=str, default='./facial_image_results2/',
                        help='')  
    
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    torch.cuda.is_available()
    
    if not os.path.exists(args.results):
        os.makedirs(args.results)
    
    
    if args.method == "2d cnn":
        args.frame_size = (28,28)
        model=CNN_2d().cuda()
        model.apply(initialize_weights)  
        main(args, model)
    if args.method == "pretrained vgg":
        args.frame_size = (224,224)
        model = alexnet()
        main(args, model)


    