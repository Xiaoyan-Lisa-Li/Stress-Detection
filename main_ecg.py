#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from facial_data_process import *
from models import *
import torchvision.transforms as transforms
from utils import *
from datetime import datetime
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset
from ecg_data_process import create_ecg_data
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

   
def predict(model, test_loader, checkpoint_path, epoch, method, path, fold, n):
    results_path = path + '1d_cnn/'
    if not os.path.exists(results_path):
            os.makedirs(results_path)    
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
    x_feat = []
    
    for i, sample_batch in enumerate(test_loader):
        x_test, y = sample_batch
        total_num += len(y)
        with torch.no_grad():
            pred, features = model(x_test.cuda())
        x_feat.extend(features.cpu().detach().numpy())
        
        pred = pred.squeeze()
        ### if preds > 0.5, preds = 1, otherwise, preds = 0       
        pred = [p.item() > 0.5 for p in pred.cpu().detach().numpy()]

        pred = list(map(int, pred))
        
        # print('true y is',y)
        # print("predicted y is",pred)

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
    
    rest_precision = rest_true/(rest_true+focus_false)
    focus_precision = focus_true/(focus_true+rest_false)    
    print("total number of samples is: ",total_num)
    acc = test_acc/total_num
    print("test accuracy is {}".format(acc))
    print('rest precision is',rest_precision)
    print('focus precision is',focus_precision)
    
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
        f.writelines("the rest precision is {} \n".format(rest_precision))
        f.writelines("the focus precision is {} \n".format(focus_precision))
        

   
    fig_path = results_path + "/1d_cnn_figures/"
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
       
    plot_confusion2(y_true, y_pred, method, fig_path, fold, n, labels = [0,1])
    
    return acc,  rest_precision, focus_precision, x_feat, y_true


def train_model(model, train_loader, test_loader, num_epochs, checkpoint, results_path,fold, n):
    
    checkpoint_path = results_path + checkpoint
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        
    # optimizer = Adam(params=model.parameters(), lr=lr, amsgrad=False)
     
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    criterion = nn.BCELoss().cuda()

    

    train_losses = []
    val_losses = []
    x_feat = []
    y_t = []
    for epoch in range(num_epochs): 
        losses = 0
        avg_loss = 0.
        val_loss = 0
        model.train()
        # print(train_loader)
        for i, sample_batch in enumerate(train_loader):
        
            x_train, y_train = sample_batch
            preds, features = model(x_train.cuda())
            optimizer.zero_grad()

            loss = criterion(preds.squeeze(), y_train.cuda().float())
            # loss = criterion(preds, y_train.cuda().long())
            
            loss.backward()
            optimizer.step()
            
            losses += loss.item()
            
            ### get extracted features from cnn
            if epoch == num_epochs-1:
                x_feat.extend(features.cpu().detach().numpy())
                y_t.extend(y_train.cpu().detach().numpy())
            
        avg_loss = losses / len(train_loader)            
        train_losses.append(loss.item())  
           
        with torch.no_grad():
            for x_val, y_val in test_loader:
                x_val = x_val.cuda()
                y_val = y_val.cuda()
                
                model.eval()
  
                yhat, _ = model(x_val)
                val_l = criterion(yhat.squeeze(), y_val.float())
                val_loss += val_l.item()
                
            val_losses.append(val_loss)
                    

        
        print('Repeat {} Fold {} Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(n, fold,
            epoch, num_epochs, i, len(train_loader), loss.item()))
        
        
        if epoch == num_epochs-1:
            save_path = checkpoint_path + 'model{}_repeat{}_fold{}.pth'.format(epoch,n,fold)  
            save_objs(model, epoch, avg_loss, optimizer, save_path)
        
        if epoch % 100 == 0:
            plot_loss(args.method, train_losses, val_losses, results_path)
    
    return x_feat, y_t

def svm_rf_xgb_f(results_srx, results_f, method, x_train, y_train, x_test, y_test, fold, n):
      
    if method == 'svm':
        # model= svm.SVC(kernel='poly')
        model = svm.SVC(kernel='rbf')
    if method == 'rf':
        model = RandomForestClassifier(n_estimators=300, random_state=0)
    if method == 'xgb':
        model = xgb.XGBClassifier(objective="binary:logistic", learning_rate=0.1, max_depth=30, n_estimators=300, random_state=42)
    
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)

    class_names = ['rest', 'focus']
    
    rest_true = 0
    rest_false = 0
    focus_true = 0
    focus_false = 0  
    
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)
    
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    
    # print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")
  
    ## calculate how many samples are predicted correctly.
    for t, p in zip(y_test, y_pred):
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
    acc = accuracy_score(y_test, y_pred)
    
    
    print("accuracy is %0.4f \n" % (acc))   
    print("rest has %0.4f precision \n" % (rest_precision)) 
    print("focus has %0.4f precision\n" % (focus_precision))

    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    with open(results_f, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        f.writelines("repeat {}, fold {} \n".format(n, fold))
        f.writelines("accuracy is %0.4f \n" % (acc))    
        f.writelines("rest has %0.4f precision \n" % (rest_precision))
        f.writelines("focus has %0.4f precision\n" % (focus_precision))
     
    plot_confusion(model, x_test, y_test, class_names,results_srx, method, n, fold)
    
    return acc, rest_precision, focus_precision

def cnn_ecg(args):
    checkpoint = 'check_point/'

    k_folds = 5
    repeat = 3
    time_s = 360
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    x,y = create_ecg_data(time_s, window_s=3)
    tensor_x = torch.Tensor(x) 
    tensor_y = torch.Tensor(y)
    dataset = TensorDataset(tensor_x,tensor_y)
    
    test_acc = []
    rest_prec_ls = []
    focus_prec_ls = []
    
    test_acc_cnn = []
    rest_prec_cnn = []
    focus_prec_cnn = []
    
    for n in range(repeat):
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):          
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
            xtr_feat, y_train = train_model(model, train_loader, test_loader, args.num_epochs, checkpoint, args.results, fold, n)


            if args.add_model:
                method = 'xgb'
                path_comb = args.results + '{}_{}/'.format(args.method, method)
                if not os.path.exists(path_comb):
                    os.makedirs(path_comb)
                 ### predict cnn model
                acc_cnn, rest_precision_cnn, focus_precision_cnn, xte_feat, y_test =\
                    predict(model, test_loader, checkpoint_path, args.num_epochs-1, args.method, path_comb, fold, n)
            else:
                acc_cnn, rest_precision_cnn, focus_precision_cnn, xte_feat, y_test =\
                    predict(model, test_loader, checkpoint_path, args.num_epochs-1, args.method, args.results, fold, n)
                
            test_acc_cnn.append(acc_cnn)
            rest_prec_cnn.append(rest_precision_cnn)
            focus_prec_cnn.append(focus_precision_cnn) 
            

            if args.add_model:
                
                results_f = path_comb + '{}_{}_restults.txt'.format(args.method,method)
  
                acc, rest_precision, focus_precision =  \
                    svm_rf_xgb_f(path_comb, results_f, method, np.array(xtr_feat), np.array(y_train), np.array(xte_feat), np.array(y_test), fold, n)
                                    
                test_acc.append(acc)
                focus_prec_ls.append(focus_precision)
                rest_prec_ls.append(rest_precision)
            else:
                results_f = args.results + '{}_restults.txt'.format(args.method)
            
    if args.add_model:    
        ### results of combined mothod        
        acc_mean = np.array(test_acc).mean()
        acc_std = np.array(test_acc).std()
    
        rest_mean = np.array(rest_prec_ls).mean()
        rest_std = np.array(rest_prec_ls).std()
        
        focus_mean = np.array(focus_prec_ls).mean()
        focus_std = np.array(focus_prec_ls).std()
        
        print("Method %s_%s: %0.4f accuracy with a standard deviation of %0.4f \n" % (args.method, method, acc_mean, acc_std))    
        print("Method %s_%s: %0.4f rest precision with a standard deviation of %0.4f \n" % (args.method,method, rest_mean, rest_std))
        print("Method %s_%s: %0.4f focus with a standard deviation of %0.4f \n" % (args.method,method, focus_mean, focus_std))
            
        
    ### results of cnn
    acc_mean_cnn = np.array(test_acc_cnn).mean()
    acc_std_cnn = np.array(test_acc_cnn).std()

    rest_mean_cnn = np.array(rest_prec_cnn).mean()
    rest_std_cnn = np.array(rest_prec_cnn).std()
    
    focus_mean_cnn = np.array(focus_prec_cnn).mean()
    focus_std_cnn = np.array(focus_prec_cnn).std()    

    print("Method %s: %0.4f accuracy with a standard deviation of %0.4f \n" % (args.method, acc_mean_cnn, acc_std_cnn))    
    print("Method %s: %0.4f rest precision with a standard deviation of %0.4f \n" % (args.method, rest_mean_cnn, rest_std_cnn))
    print("Method %s: %0.4f focus with a standard deviation of %0.4f \n" % (args.method, focus_mean_cnn, focus_std_cnn))     

    
    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        
    with open(results_f, "a") as f:
        if args.add_model:
            f.write('*'*40 + date_time + '*'*40 +'\n')
            f.write("Method %s_%s: %0.4f accuracy with a standard deviation of %0.4f \n" % (args.method, method, acc_mean, acc_std))    
            f.write("Method %s_%s: %0.4f rest precision with a standard deviation of %0.4f \n" % (args.method,method, rest_mean, rest_std))
            f.write("Method %s_%s: %0.4f focus with a standard deviation of %0.4f \n" % (args.method,method, focus_mean, focus_std))
        f.write('--'*40 +'\n')
        f.write("Method %s: %0.4f accuracy with a standard deviation of %0.4f \n" % (args.method, acc_mean_cnn, acc_std_cnn))    
        f.write("Method %s: %0.4f rest precision with a standard deviation of %0.4f \n" % (args.method, rest_mean_cnn, rest_std_cnn))
        f.write("Method %s: %0.4f focus with a standard deviation of %0.4f \n" % (args.method, focus_mean_cnn, focus_std_cnn))        
          
    
     
    
def vgg_ecg_img(args):
    method = 'pretrain_vgg'
    results = args.results + 'ecg_img/'
    if not os.path.exists(results):
        os.makedirs(results)
    image_path = './data/ecg_img/'
    image_dir = image_path + 'images/'    
    img_csv = 'image.csv'
    batch_size = 256
    num_epochs = 350
   
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])    
    
    # creat_csv(img_csv, image_path,image_dir)    
    _, _, dataset = create_datasets(batch_size, transform, image_path, image_dir, img_csv)
    checkpoint = 'check_point/'

    k_folds = 5
    repeat = 3
    time_s = 360
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    test_acc = []
    rest_prec_ls = []
    focus_prec_ls = []
    
    for n in range(repeat):
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):          
            print(f'FOLD {fold}')
            print('--------------------------------')  
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            
            # Define data loaders for training and testing data in this fold
            train_loader = torch.utils.data.DataLoader(
                              dataset, batch_size=batch_size, sampler=train_subsampler)
            test_loader = torch.utils.data.DataLoader(
                              dataset, batch_size=batch_size, sampler=test_subsampler)

            model = alexnet().cuda()
            reset_weights_vgg(model)

            checkpoint_path = results + checkpoint
            
            # train model
            train_model(model, train_loader, test_loader, num_epochs, checkpoint, results, fold, n)

            acc, rest_precision, focus_precision= predict(model, test_loader, checkpoint_path, num_epochs-1, method, results, fold, n)
            test_acc.append(acc)

            focus_prec_ls.append(focus_precision)
            rest_prec_ls.append(rest_precision)
            
    acc_mean = np.array(test_acc).mean()
    acc_std = np.array(test_acc).std()

    rest_mean = np.array(rest_prec_ls).mean()
    rest_std = np.array(rest_prec_ls).std()
    
    focus_mean = np.array(focus_prec_ls).mean()
    focus_std = np.array(focus_prec_ls).std()
    
    print("Method %s: %0.4f accuracy with a standard deviation of %0.4f" % (method, acc_mean, acc_std))
    print("Method %s: %0.4f rest precision with a standard deviation of %0.4f" % (method, rest_mean, rest_std))
    print("Method %s: %0.4f focus precision with a standard deviation of %0.4f" % (method, focus_mean, focus_std))
    
    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    results_f = results + '{}_restults.txt'.format(method)
    with open(results_f, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        f.write("Method %s: %0.4f accuracy with a standard deviation of %0.4f \n" % (method, acc_mean, acc_std))      
        f.write("Method %s: %0.4f rest precision with a standard deviation of %0.4f \n" % (method, rest_mean, rest_std))
        f.write("Method %s: %0.4f focus precision with a standard deviation of %0.4f \n" % (method, focus_mean, focus_std))
        
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_epochs', type=int, default=1500,
                        help='')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='')
    parser.add_argument('--method', type=str, default='1d_cnn',
                        help='')   
    parser.add_argument('--add_model', type=bool, default=True,
                        help='')      
    parser.add_argument('--results', type=str, default='./ecg_results/',
                        help='')  
    
    args = parser.parse_args()
    
    # seed_everything(args.seed)
    
    torch.cuda.is_available()
    
    if not os.path.exists(args.results):
        os.makedirs(args.results)
    
    # cnn_ecg(args)
    vgg_ecg_img(args)
    
