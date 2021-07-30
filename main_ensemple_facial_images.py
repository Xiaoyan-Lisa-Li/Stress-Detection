#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time 
import random
from torch.optim import Adam
from facial_data_process import *
from models import *
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
from utils import *
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import xgboost as xgb
import scipy.stats as st
import pickle
import sys
from ecg_data_process import create_ecg_data

   
def predict_img(model, test_loader, epoch, method, path, fold, n):
    
    results_path = path + '{}/'.format(method)
    if not os.path.exists(results_path):
            os.makedirs(results_path)  
            
    results_f = results_path + '{}_restults.txt'.format(method)    
        
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
    y_prob = []

    for x_test, test_y in test_loader:
        total_num += len(test_y)
        with torch.no_grad():
            prob, _ = model(x_test.cuda())
        
        prob = prob.cpu().detach().numpy()       
        ### the output probability
        y_prob.extend([p.item() for p in prob])
                
        ### if preds > 0.5, preds = 1, otherwise, preds = 0
        pred = [p.item() > 0.5 for p in prob]
        y_pred.extend(list(map(int, pred)))
        
        test_y = test_y.detach().numpy()    
        y_true.extend(test_y.tolist())
        
        test_acc += sum(test_y == np.array(pred))
        
       
        ## calculate how many samples are predicted correctly.
        for t, p in zip(test_y, pred):
            if t == p and t.item() == 0:
                rest_true += 1
            elif t != p and t.item() == 0:
                focus_false += 1
            elif t == p and t.item() == 1:
                focus_true += 1
            else:
                rest_false += 1

    rest_precision = rest_true/(rest_true+rest_false)
    focus_precision = focus_true/(focus_true+focus_false)
    rest_recall = rest_true/(rest_true+focus_false)
    focus_recall = focus_true/(focus_true+rest_false)
    
    print("This is the {}th repeat {}th fold".format(n, fold))            
    print("rest_true is ",rest_true)  
    print("focus_false is ",focus_false)  
    print("focus_true is ",focus_true)    
    print("rest_false is ",rest_false)

    print('rest precision is',rest_precision)
    print('focus precision is',focus_precision)
    print('rest recall is',rest_recall)
    print('focus recall is',focus_recall)
    
    print("total number of samples is: ",total_num)
    acc = test_acc.item()/total_num
    print("test accuracy is {}".format(acc))
    
    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    
    with open(results_f, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        f.writelines("repeat {}, fold {} \n".format(n, fold))
        f.writelines("the number of rest samples that are correctly classified is {} \n".format(rest_true))
        f.writelines("the number of rest samples that are incorrectly classified is {} \n".format(focus_false))
        f.writelines("the number of focus samples that are correctly classified is {} \n".format(focus_true))
        f.writelines("the number of focus samples that are incorrectly classified is {} \n".format(rest_false))

        f.writelines("the rest precision is {} \n".format(rest_precision))
        f.writelines("the focus precision is {} \n".format(focus_precision))
        f.writelines("the rest recall is {} \n".format(rest_recall))
        f.writelines("the focus recall is {} \n".format(focus_recall))
        
        f.writelines("The test accracy of {} is {} \n".format(method, acc))
    
            
    fig_path = results_path + "/{}_figures/".format(method)     
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)            
                
    plot_confusion2(y_true, y_pred, method, fig_path, fold, n, labels = [0,1])
    
    return  y_true, y_prob, acc, rest_precision, focus_precision, rest_recall, focus_recall

def predict_svm_xgb(model, x_test, y_test, path, method, fold, n):
    rest_true = 0
    rest_false = 0
    focus_true = 0
    focus_false = 0  
    focus_pre_ls = []
    rest_pre_ls = []
    acc_ls = []
    
    results_path = path + '{}/'.format(method)   
    if not os.path.exists(results_path):
            os.makedirs(results_path) 
    results_f = results_path + '{}_restults.txt'.format(method)
    
    y_pred = model.predict_proba(x_test)
    y_prob = y_pred[:,1]
    y_pred = np.argmax(y_pred, axis=1).tolist()
    print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")
  
    ## calculate how many samples are predicted correctly.
    for t, p in zip(y_test, y_pred):
        if t == p and t.item() == 0:
            rest_true += 1
        elif t != p and t.item() == 0:
            focus_false += 1
        elif t == p and t.item() == 1:
            focus_true += 1
        else:
            rest_false += 1
            
    rest_precision = rest_true/(rest_true+rest_false)
    focus_precision = focus_true/(focus_true+focus_false)
    rest_recall = rest_true/(rest_true+focus_false)
    focus_recall = focus_true/(focus_true+rest_false)

    
    acc = accuracy_score(y_test, y_pred)
    
    rest_pre_ls.append(rest_precision)
    focus_pre_ls.append(focus_precision)
    acc_ls.append(acc)
    
    print("accuracy is %0.4f \n" % (acc))
    print("focus has %0.4f precision\n" % (focus_precision))
    print("rest has %0.4f precision \n" % (rest_precision))  
    print("focus has %0.4f recall\n" % (focus_recall))
    print("rest has %0.4f recall \n" % (rest_recall))     

    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    with open(results_f, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        f.writelines("the number of rest samples that are correctly classified is {} \n".format(rest_true))
        f.writelines("the number of rest samples that are incorrectly classified is {} \n".format(focus_false))
        f.writelines("the number of focus samples that are correctly classified is {} \n".format(focus_true))
        f.writelines("the number of focus samples that are incorrectly classified is {} \n".format(rest_false))
        f.writelines("the rest precision is {} \n".format(rest_precision))
        f.writelines("the focus precision is {} \n".format(focus_precision))
        f.writelines("the rest recall is {} \n".format(rest_recall))
        f.writelines("the focus recall is {} \n".format(focus_recall))       
        f.writelines("The test accracy of {} is {} \n".format(method, acc))
        
    class_names = ['rest', 'focus']  
    plot_confusion(model, x_test, y_test, class_names, results_path, method, fold, n)
 
    
    return y_prob, acc, rest_precision, focus_precision, rest_recall, focus_recall

def train_model(model, train_loader, test_loader, num_epochs, results_path, method, fold, n):
      
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    criterion = nn.BCELoss().cuda()

    train_losses = []
    val_losses = []
    for epoch in range(num_epochs): 
        losses = 0
        avg_loss = 0.
        val_loss = 0.
        model.train()
        # print(train_loader)
        for i, sample_batch in enumerate(train_loader):
        
            x_train, y_train = sample_batch
            preds, _ = model(x_train.cuda())
            optimizer.zero_grad()

            loss = criterion(preds.squeeze(), y_train.cuda().float())
            
            loss.backward()
            optimizer.step()
            
            losses += loss.item()
            
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
        

    
    plot_loss(method, train_losses, val_losses, results_path)
    
    

def main(args,results_face, results_ecg):
    
    ###########################################################################
    ### create dataset for facial image
    image_path = './data/images_224x224/'
    
    image_dir = image_path + 'images/'
    img_csv = 'image.csv'
    
    
    k_folds = 5
    repeat = 3
    
      # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    _, _, dataset_224 = create_datasets(args.batch_size,transform, image_path, image_dir, img_csv)
     
    acc_ensem_fac = []
    acc_vgg_fac = []
    acc_svm_fac = []
    acc_xgb_fac = []
    
    rest_pre_ensem_fac = []
    focus_pre_ensem_fac = []
    rest_pre_vgg_fac = []
    focus_pre_vgg_fac = []
    rest_pre_svm_fac = []
    focus_pre_svm_fac = []
    rest_pre_xgb_fac = []
    focus_pre_xgb_fac = []

    rest_rec_ensem_fac = []
    focus_rec_ensem_fac = []
    rest_rec_vgg_fac = []
    focus_rec_vgg_fac = []
    rest_rec_svm_fac = []
    focus_rec_svm_fac = []
    rest_rec_xgb_fac = []
    focus_rec_xgb_fac = []    
    
    
    rest_true_f = 0
    rest_false_f = 0
    focus_true_f = 0
    focus_false_f = 0
    
    
    x_224 = []
    y_224 = []    
    for i in range(len(dataset_224)):    
        x_224.append(dataset_224[i][0].numpy().flatten())
        y_224.append(dataset_224[i][1].numpy())
    x_224 = np.asarray(x_224)
    y_224 = np.asarray(y_224)

    for n in range(repeat):
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset_224)):
            print(f'FOLD {fold}')
            ###################################################################
            ### use facial images to train vgg, svm and xgboost
            ###################################################################
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            ### create trainset and testset for pretained vgg
            train_loader_224 = torch.utils.data.DataLoader(
                              dataset_224, batch_size=args.batch_size, sampler=train_subsampler)
            
            testset = torch.utils.data.Subset(dataset_224, test_ids.tolist())
            test_loader_224 = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False) 
            
            ### create train set and test set for svm
            x_train_224, y_train_224 = x_224[train_ids], y_224[train_ids]
            x_test_224, y_test_224 = x_224[test_ids], y_224[test_ids]           
            x_train_224, y_train_224 = shuffle(x_train_224, y_train_224) ## only shuffle train dataset

            
            method1 = 'vgg_face'
            vgg_face = alexnet().cuda()
            reset_weights_vgg(vgg_face)
            num_epochs = 150
            
            start_vgg = datetime.now() 
            train_model(vgg_face, train_loader_224, test_loader_224, num_epochs, results_face, method1, fold, n)
            y_vgg_f, pred_vgg_f, acc_vgg_f, rest_pre_vgg_f, focus_pre_vgg_f, rest_rec_vgg_f, focus_rec_vgg_f =\
                predict_img(vgg_face, test_loader_224, num_epochs-1, method1, results_face, fold, n)
            
            ### calculate the running time of the model
            run_time_vgg = datetime.now() - start_vgg
            print('the training time of method {} is {}'.format(method1, run_time_vgg))
            ### get the size of the model
            p = pickle.dumps(vgg_face)
            vgg_size = sys.getsizeof(p)
            print("the size of method {} is {}".format(method1,vgg_size ))
            
            
            method2 = 'svm_face'
            svm_face= svm.SVC(kernel='poly', probability=True)
            # svm_face= svm.SVC(kernel='rbf', C=5.0, probability=True)
            start_svm = datetime.now() 
            svm_face.fit(x_train_224,y_train_224)
            pred_svm_f, acc_svm_f, rest_pre_svm_f, focus_pre_svm_f, rest_rec_svm_f, focus_rec_svm_f = \
                predict_svm_xgb(svm_face, x_test_224, y_test_224, results_face, method2, fold, n)            
            ### calculate the running time of the model
            run_time_svm= datetime.now() - start_svm
            print('the training time of method {} is {}'.format(method2, run_time_svm))   
            ### get the size of the model
            p = pickle.dumps(svm_face)
            svm_size = sys.getsizeof(p)
            print("the size of method {} is {}".format(method2,svm_size))
  
            
            method3 = 'xgb_face'
            xgb_face= xgb.XGBClassifier(learning_rate =0.1, n_estimators=200, max_depth=5, min_child_weight=1, gamma=0,  subsample=0.8,\
                                  colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)  
                
            start_xgb = datetime.now()    
            xgb_face.fit(x_train_224,y_train_224)
            pred_xgb_f, acc_xgb_f, rest_pre_xgb_f, focus_pre_xgb_f, rest_rec_xgb_f, focus_rec_xgb_f = \
                predict_svm_xgb(xgb_face, x_test_224, y_test_224, results_face, method3, fold, n)
            ### calculate the running time of the model
            run_time_xgb= datetime.now() - start_xgb
            print('the training time of method {} is {}'.format(method3, run_time_xgb))   
            ### get the size of the model
            p = pickle.dumps(xgb_face)
            xgb_size = sys.getsizeof(p)
            print("the size of method {} is {}".format(method3,xgb_size))            

            print('y_test_224 == y_vgg ?', y_test_224==y_vgg_f)

            y_ensem_f = []
            for i in range(len(y_test_224)):
                pred_ave_ensem = (pred_vgg_f[i] + pred_svm_f[i] + pred_xgb_f[i])/3
                if pred_ave_ensem >= 0.5:
                    y_ensem_f.append(1)
                else:
                    y_ensem_f.append(0)
                
            
            ## calculate how many samples are predicted correctly.
            for t, e in zip(y_test_224, y_ensem_f):
                if t == e and t.item() == 0:
                    rest_true_f += 1
                elif t != e and t.item() == 0:
                    focus_false_f += 1
                elif t == e and t.item() == 1:
                    focus_true_f += 1
                else:
                    rest_false_f += 1
        
            rest_pre_ensem_f = rest_true_f/(rest_true_f+rest_false_f)
            focus_pre_ensem_f = focus_true_f/(focus_true_f+focus_false_f)
            
            rest_rec_ensem_f = rest_true_f/(rest_true_f+focus_false_f)
            focus_rec_ensem_f = focus_true_f/(focus_true_f+rest_false_f)
            
            acc_ensem_f = accuracy_score(y_test_224, y_ensem_f)

    
            rest_pre_ensem_fac.append(rest_pre_ensem_f)
            focus_pre_ensem_fac.append(focus_pre_ensem_f)
            rest_pre_vgg_fac.append(rest_pre_vgg_f)
            focus_pre_vgg_fac.append(focus_pre_vgg_f)
            rest_pre_svm_fac.append(rest_pre_svm_f)
            focus_pre_svm_fac.append(focus_pre_svm_f)            
            rest_pre_xgb_fac.append(rest_pre_xgb_f)
            focus_pre_xgb_fac.append(focus_pre_xgb_f)  
            
            rest_rec_ensem_fac.append(rest_rec_ensem_f)
            focus_rec_ensem_fac.append(focus_rec_ensem_f)
            rest_rec_vgg_fac.append(rest_rec_vgg_f)
            focus_rec_vgg_fac.append(focus_rec_vgg_f)
            rest_rec_svm_fac.append(rest_rec_svm_f)
            focus_rec_svm_fac.append(focus_rec_svm_f)            
            rest_rec_xgb_fac.append(rest_rec_xgb_f)
            focus_rec_xgb_fac.append(focus_rec_xgb_f)  
            
            acc_ensem_fac.append(acc_ensem_f)
            acc_vgg_fac.append(acc_vgg_f)
            acc_svm_fac.append(acc_svm_f)
            acc_xgb_fac.append(acc_xgb_f)
                                           
            fig_path = results_face + "/emsemble_figures/"
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            plot_confusion2(y_vgg_f, y_ensem_f, args.method, fig_path, fold, n, labels = [0,1])              

    
    ###########################################################################
    #### results of facial images based methods
    ### results of accuracy
    acc_ensem_mean_f = np.array(acc_ensem_fac).mean()
    acc_ensem_std_f = np.array(acc_ensem_fac).std()
    acc_ensem_conf_f = st.t.interval(0.95, len(acc_ensem_fac)-1, loc=np.mean(acc_ensem_fac), scale=st.sem(acc_ensem_fac))
    acc_vgg_mean_f = np.array(acc_vgg_fac).mean()
    acc_vgg_std_f = np.array(acc_vgg_fac).std()
    acc_vgg_conf_f = st.t.interval(0.95, len(acc_vgg_fac)-1, loc=np.mean(acc_vgg_fac), scale=st.sem(acc_vgg_fac))
    acc_svm_mean_f = np.array(acc_svm_fac).mean()
    acc_svm_std_f = np.array(acc_svm_fac).std()
    acc_svm_conf_f = st.t.interval(0.95, len(acc_svm_fac)-1, loc=np.mean(acc_svm_fac), scale=st.sem(acc_svm_fac))
    acc_xgb_mean_f = np.array(acc_xgb_fac).mean()
    acc_xgb_std_f = np.array(acc_xgb_fac).std()
    acc_xgb_conf_f = st.t.interval(0.95, len(acc_xgb_fac)-1, loc=np.mean(acc_xgb_fac), scale=st.sem(acc_xgb_fac))

    ### results of precision
    rest_pre_ensem_mean_f = np.array(rest_pre_ensem_fac).mean()
    rest_pre_ensem_std_f = np.array(rest_pre_ensem_fac).std() 
    rest_pre_ensem_conf_f = st.t.interval(0.95, len(rest_pre_ensem_fac)-1, loc=np.mean(rest_pre_ensem_fac), scale=st.sem(rest_pre_ensem_fac))
    focus_pre_ensem_mean_f = np.array(focus_pre_ensem_fac).mean()
    focus_pre_ensem_std_f = np.array(focus_pre_ensem_fac).std()
    focus_pre_ensem_conf_f = st.t.interval(0.95, len(focus_pre_ensem_fac)-1, loc=np.mean(focus_pre_ensem_fac), scale=st.sem(focus_pre_ensem_fac))
    
    rest_pre_vgg_mean_f = np.array(rest_pre_vgg_fac).mean()
    rest_pre_vgg_std_f = np.array(rest_pre_vgg_fac).std() 
    rest_pre_vgg_conf_f = st.t.interval(0.95, len(rest_pre_vgg_fac)-1, loc=np.mean(rest_pre_vgg_fac), scale=st.sem(rest_pre_vgg_fac))
    focus_pre_vgg_mean_f = np.array(focus_pre_vgg_fac).mean()
    focus_pre_vgg_std_f = np.array(focus_pre_vgg_fac).std()
    focus_pre_vgg_conf_f = st.t.interval(0.95, len(focus_pre_vgg_fac)-1, loc=np.mean(focus_pre_vgg_fac), scale=st.sem(focus_pre_vgg_fac))

    rest_pre_svm_mean_f = np.array(rest_pre_svm_fac).mean()
    rest_pre_svm_std_f = np.array(rest_pre_svm_fac).std()  
    rest_pre_svm_conf_f = st.t.interval(0.95, len(rest_pre_svm_fac)-1, loc=np.mean(rest_pre_svm_fac), scale=st.sem(rest_pre_svm_fac))
    focus_pre_svm_mean_f = np.array(focus_pre_svm_fac).mean()
    focus_pre_svm_std_f = np.array(focus_pre_svm_fac).std() 
    focus_pre_svm_conf_f = st.t.interval(0.95, len(focus_pre_svm_fac)-1, loc=np.mean(focus_pre_svm_fac), scale=st.sem(focus_pre_svm_fac))
    
    rest_pre_xgb_mean_f = np.array(rest_pre_xgb_fac).mean()
    rest_pre_xgb_std_f = np.array(rest_pre_xgb_fac).std() 
    rest_pre_xgb_conf_f = st.t.interval(0.95, len(rest_pre_xgb_fac)-1, loc=np.mean(rest_pre_xgb_fac), scale=st.sem(rest_pre_xgb_fac))
    focus_pre_xgb_mean_f = np.array(focus_pre_xgb_fac).mean()
    focus_pre_xgb_std_f = np.array(focus_pre_xgb_fac).std()
    focus_pre_xgb_conf_f = st.t.interval(0.95, len(focus_pre_xgb_fac)-1, loc=np.mean(focus_pre_xgb_fac), scale=st.sem(focus_pre_xgb_fac))
    
    ### results of recall
    rest_rec_ensem_mean_f = np.array(rest_rec_ensem_fac).mean()
    rest_rec_ensem_std_f = np.array(rest_rec_ensem_fac).std() 
    rest_rec_ensem_conf_f = st.t.interval(0.95, len(rest_rec_ensem_fac)-1, loc=np.mean(rest_rec_ensem_fac), scale=st.sem(rest_rec_ensem_fac))
    focus_rec_ensem_mean_f = np.array(focus_rec_ensem_fac).mean()
    focus_rec_ensem_std_f = np.array(focus_rec_ensem_fac).std()
    focus_rec_ensem_conf_f = st.t.interval(0.95, len(focus_rec_ensem_fac)-1, loc=np.mean(focus_rec_ensem_fac), scale=st.sem(focus_rec_ensem_fac))
    
    rest_rec_vgg_mean_f = np.array(rest_rec_vgg_fac).mean()
    rest_rec_vgg_std_f = np.array(rest_rec_vgg_fac).std() 
    rest_rec_vgg_conf_f = st.t.interval(0.95, len(rest_rec_vgg_fac)-1, loc=np.mean(rest_rec_vgg_fac), scale=st.sem(rest_rec_vgg_fac))
    focus_rec_vgg_mean_f = np.array(focus_rec_vgg_fac).mean()
    focus_rec_vgg_std_f = np.array(focus_rec_vgg_fac).std()
    focus_rec_vgg_conf_f = st.t.interval(0.95, len(focus_rec_vgg_fac)-1, loc=np.mean(focus_rec_vgg_fac), scale=st.sem(focus_rec_vgg_fac))

    rest_rec_svm_mean_f = np.array(rest_rec_svm_fac).mean()
    rest_rec_svm_std_f = np.array(rest_rec_svm_fac).std()  
    rest_rec_svm_conf_f = st.t.interval(0.95, len(rest_rec_svm_fac)-1, loc=np.mean(rest_rec_svm_fac), scale=st.sem(rest_rec_svm_fac))
    focus_rec_svm_mean_f = np.array(focus_rec_svm_fac).mean()
    focus_rec_svm_std_f = np.array(focus_rec_svm_fac).std() 
    focus_rec_svm_conf_f = st.t.interval(0.95, len(focus_rec_svm_fac)-1, loc=np.mean(focus_rec_svm_fac), scale=st.sem(focus_rec_svm_fac))
    
    rest_rec_xgb_mean_f = np.array(rest_rec_xgb_fac).mean()
    rest_rec_xgb_std_f = np.array(rest_rec_xgb_fac).std() 
    rest_rec_xgb_conf_f = st.t.interval(0.95, len(rest_rec_xgb_fac)-1, loc=np.mean(rest_rec_xgb_fac), scale=st.sem(rest_rec_xgb_fac))
    focus_rec_xgb_mean_f = np.array(focus_rec_xgb_fac).mean()
    focus_rec_xgb_std_f = np.array(focus_rec_xgb_fac).std()
    focus_rec_xgb_conf_f = st.t.interval(0.95, len(focus_rec_xgb_fac)-1, loc=np.mean(focus_rec_xgb_fac), scale=st.sem(focus_rec_xgb_fac))    
    
   
    
    ### print accuracy
    print("facial image method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, acc_ensem_mean_f, acc_ensem_std_f,acc_ensem_conf_f[0],acc_ensem_conf_f[1]))
    print("facial image method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method1, acc_vgg_mean_f, acc_vgg_std_f, acc_vgg_conf_f[0],acc_vgg_conf_f[1]))
    print("facial image method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method2, acc_svm_mean_f, acc_svm_std_f, acc_svm_conf_f[0],acc_svm_conf_f[1]))
    print("facial image method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method3, acc_xgb_mean_f, acc_xgb_std_f, acc_xgb_conf_f[0],acc_xgb_conf_f[1]))
    
    ### print precision
    print("facial image method %s: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_pre_ensem_mean_f, rest_pre_ensem_std_f, rest_pre_ensem_conf_f[0], rest_pre_ensem_conf_f[1]))
    print("facial image method %s: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_pre_ensem_mean_f, focus_pre_ensem_std_f, focus_pre_ensem_conf_f[0],focus_pre_ensem_conf_f[1]))
    print("facial image method vgg: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_vgg_mean_f, rest_pre_vgg_std_f, rest_pre_vgg_conf_f[0],rest_pre_vgg_conf_f[1]))
    print("facial image method vgg: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_vgg_mean_f, focus_pre_vgg_std_f, focus_pre_vgg_conf_f[0], focus_pre_vgg_conf_f[1]))
    print("facial image method svm: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_svm_mean_f, rest_pre_svm_std_f, rest_pre_svm_conf_f[0], rest_pre_svm_conf_f[1]))
    print("facial image method svm: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_svm_mean_f, focus_pre_svm_std_f, focus_pre_svm_conf_f[0], focus_pre_svm_conf_f[1]))   
    print("facial image method xgb: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_xgb_mean_f, rest_pre_xgb_std_f, rest_pre_xgb_conf_f[0], rest_pre_xgb_conf_f[1]))
    print("facial image method xgb: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_xgb_mean_f, focus_pre_xgb_std_f, focus_pre_xgb_conf_f[0], focus_pre_xgb_conf_f[1]))

    ### print recall
    print("facial image method %s: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_rec_ensem_mean_f, rest_rec_ensem_std_f, rest_rec_ensem_conf_f[0], rest_rec_ensem_conf_f[1]))
    print("facial image method %s: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_rec_ensem_mean_f, focus_rec_ensem_std_f, focus_rec_ensem_conf_f[0],focus_rec_ensem_conf_f[1]))
    print("facial image method vgg: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_vgg_mean_f, rest_rec_vgg_std_f, rest_rec_vgg_conf_f[0],rest_rec_vgg_conf_f[1]))
    print("facial image method vgg: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_vgg_mean_f, focus_rec_vgg_std_f, focus_rec_vgg_conf_f[0], focus_rec_vgg_conf_f[1]))
    print("facial image method svm: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_svm_mean_f, rest_rec_svm_std_f, rest_rec_svm_conf_f[0], rest_rec_svm_conf_f[1]))
    print("facial image method svm: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_svm_mean_f, focus_rec_svm_std_f, focus_rec_svm_conf_f[0], focus_rec_svm_conf_f[1]))   
    print("facial image method xgb: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_xgb_mean_f, rest_rec_xgb_std_f, rest_rec_xgb_conf_f[0], rest_rec_xgb_conf_f[1]))
    print("facial image method xgb: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_xgb_mean_f, focus_rec_xgb_std_f, focus_rec_xgb_conf_f[0], focus_rec_xgb_conf_f[1]))    

    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    results_f1 = results_face + '{}_restults.txt'.format(args.method)
    with open(results_f1, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        ### write accuracy
        f.write("facial image method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, acc_ensem_mean_f, acc_ensem_std_f,acc_ensem_conf_f[0],acc_ensem_conf_f[1]))
        f.write("facial image method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method1, acc_vgg_mean_f, acc_vgg_std_f, acc_vgg_conf_f[0],acc_vgg_conf_f[1]))
        f.write("facial image method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method2, acc_svm_mean_f, acc_svm_std_f, acc_svm_conf_f[0],acc_svm_conf_f[1]))
        f.write("facial image method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method3, acc_xgb_mean_f, acc_xgb_std_f, acc_xgb_conf_f[0],acc_xgb_conf_f[1]))
        f.write('--'*40 +'\n')
        ### write precision
        f.write("facial image method %s: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_pre_ensem_mean_f, rest_pre_ensem_std_f, rest_pre_ensem_conf_f[0], rest_pre_ensem_conf_f[1]))
        f.write("facial image method %s: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_pre_ensem_mean_f, focus_pre_ensem_std_f, focus_pre_ensem_conf_f[0],focus_pre_ensem_conf_f[1]))
        f.write("facial image method vgg: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_vgg_mean_f, rest_pre_vgg_std_f, rest_pre_vgg_conf_f[0],rest_pre_vgg_conf_f[1]))
        f.write("facial image method vgg: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_vgg_mean_f, focus_pre_vgg_std_f, focus_pre_vgg_conf_f[0], focus_pre_vgg_conf_f[1]))
        f.write("facial image method svm: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_svm_mean_f, rest_pre_svm_std_f, rest_pre_svm_conf_f[0], rest_pre_svm_conf_f[1]))
        f.write("facial image method svm: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_svm_mean_f, focus_pre_svm_std_f, focus_pre_svm_conf_f[0], focus_pre_svm_conf_f[1]))   
        f.write("facial image method xgb: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_xgb_mean_f, rest_pre_xgb_std_f, rest_pre_xgb_conf_f[0], rest_pre_xgb_conf_f[1]))
        f.write("facial image method xgb: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_xgb_mean_f, focus_pre_xgb_std_f, focus_pre_xgb_conf_f[0], focus_pre_xgb_conf_f[1]))

        f.write('--'*40 +'\n')
        ### write recall
        f.write("facial image method %s: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_rec_ensem_mean_f, rest_rec_ensem_std_f, rest_rec_ensem_conf_f[0], rest_rec_ensem_conf_f[1]))
        f.write("facial image method %s: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_rec_ensem_mean_f, focus_rec_ensem_std_f, focus_rec_ensem_conf_f[0],focus_rec_ensem_conf_f[1]))
        f.write("facial image method vgg: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_vgg_mean_f, rest_rec_vgg_std_f, rest_rec_vgg_conf_f[0],rest_rec_vgg_conf_f[1]))
        f.write("facial image method vgg: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_vgg_mean_f, focus_rec_vgg_std_f, focus_rec_vgg_conf_f[0], focus_rec_vgg_conf_f[1]))
        f.write("facial image method svm: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_svm_mean_f, rest_rec_svm_std_f, rest_rec_svm_conf_f[0], rest_rec_svm_conf_f[1]))
        f.write("facial image method svm: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_svm_mean_f, focus_rec_svm_std_f, focus_rec_svm_conf_f[0], focus_rec_svm_conf_f[1]))   
        f.write("facial image method xgb: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_xgb_mean_f, rest_rec_xgb_std_f, rest_rec_xgb_conf_f[0], rest_rec_xgb_conf_f[1]))
        f.write("facial image method xgb: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_xgb_mean_f, focus_rec_xgb_std_f, focus_rec_xgb_conf_f[0], focus_rec_xgb_conf_f[1]))    
        f.write('--'*40 +'\n')
        f.write('facial image based: the size of method vgg is {}, and the runing time is {} \n'.format(vgg_size, run_time_vgg))
        f.write('facial image based: the size of method svm is {}, and the runing time is {} \n'.format(svm_size, run_time_svm))
        f.write('facial image based: the size of method xgb is {}, and the runing time is {} \n'.format(xgb_size, run_time_xgb))

    


    
    
                
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='')
    parser.add_argument('--num_epochs', type=int, default=150,
                        help='')
    parser.add_argument('--method', type=str, default='ensemble',
                        help='')      
    parser.add_argument('--results', type=str, default='./ensemble_results_6mins/',
                        help='')  
    
    args = parser.parse_args()
    
    # seed_everything(args.seed)
    
    torch.cuda.is_available()
    
    results_face = args.results+'facial_image/'
    results_ecg = args.results+'ecg/'
    if not os.path.exists(results_face):
        os.makedirs(results_face)
    if not os.path.exists(results_ecg):
        os.makedirs(results_ecg)    

    main(args, results_face, results_ecg)

    
