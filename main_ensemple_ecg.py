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
        f.writelines("accuracy is %0.4f \n" % (acc))

        f.writelines("the number of rest samples that are correctly classified is {} \n".format(rest_true))
        f.writelines("the number of rest samples that are incorrectly classified is {} \n".format(focus_false))
        f.writelines("the number of focus samples that are correctly classified is {} \n".format(focus_true))
        f.writelines("the number of focus samples that are incorrectly classified is {} \n".format(rest_false))
        f.writelines("focus has %0.4f precision\n" % (focus_precision))
        f.writelines("rest has %0.4f precision \n" % (rest_precision))
        f.writelines("focus has %0.4f recall \n" % (focus_recall))
        f.writelines("rest has %0.4f recall \n" % (rest_recall))        
        
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
    
    # check_path_cnn = args.results + 'checkpt_cnn'
    # check_path_vgg = args.results + 'checkpt_vgg'
    
    k_folds = 5
    repeat = 3
    
      # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

   
    ###########################################################################
    ### create dataset for ecg signals  
    ### ecg image dataset
    image_path_ecg = './data/ecg_img/'
    image_dir_ecg = image_path_ecg + 'images/'  
    batch_size_ecg = 256
    _, _, dataset_ecg_img = create_datasets(batch_size_ecg, transform, image_path_ecg, image_dir_ecg, img_csv)

    
    ### ecg signals
    x_ecg,y_ecg = create_ecg_data(time_s=360, window_s=3)
    tensor_x_ecg = torch.Tensor(x_ecg) 
    tensor_y_ecg = torch.Tensor(y_ecg)
    dataset_ecg = TensorDataset(tensor_x_ecg,tensor_y_ecg)

    acc_ensem_ecg = []
    acc_vgg_ecg = []
    acc_cnn_ecg = []    
    acc_svm_ecg = []
    acc_xgb_ecg = []
    
    rest_pre_ensem_ecg = []
    focus_pre_ensem_ecg = []
    rest_pre_vgg_ecg = []
    focus_pre_vgg_ecg = []
    rest_pre_cnn_ecg = []
    focus_pre_cnn_ecg = []
    rest_pre_svm_ecg = []
    focus_pre_svm_ecg = []
    rest_pre_xgb_ecg = []
    focus_pre_xgb_ecg = []

    rest_rec_ensem_ecg = []
    focus_rec_ensem_ecg = []
    rest_rec_vgg_ecg = []
    focus_rec_vgg_ecg = []
    rest_rec_cnn_ecg = []
    focus_rec_cnn_ecg = []
    rest_rec_svm_ecg = []
    focus_rec_svm_ecg = []
    rest_rec_xgb_ecg = []
    focus_rec_xgb_ecg = []    
    
    
    rest_true_ecg = 0
    rest_false_ecg = 0
    focus_true_ecg = 0
    focus_false_ecg = 0
        
    
    
    for n in range(repeat):
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset_224)):
            print(f'FOLD {fold}')
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            ###################################################################
            ### use ecg signals to train 1d_cnn, vgg, svm and xgboost
            ###################################################################
            
            ### create train set and test set for pretained vgg
            train_loader_ecg_img = torch.utils.data.DataLoader(
                              dataset_ecg_img, batch_size=batch_size_ecg, sampler=train_subsampler)
            
            testset_ecg = torch.utils.data.Subset(dataset_ecg_img, test_ids.tolist())
            test_loader_ecg_img = torch.utils.data.DataLoader(testset_ecg, batch_size=batch_size_ecg, shuffle=False) 
            
                     
            method4 = 'vgg_ecg'
            vgg_ecg = alexnet().cuda()
            reset_weights_vgg(vgg_ecg)
    
            num_epochs = 350
            start_vgg_ecg = datetime.now() 
            train_model(vgg_ecg, train_loader_ecg_img, test_loader_ecg_img, num_epochs, results_ecg, method4, fold, n)
            y_vgg_e, pred_vgg_e, acc_vgg_e, rest_pre_vgg_e, focus_pre_vgg_e, rest_rec_vgg_e, focus_rec_vgg_e =\
                predict_img(vgg_ecg, test_loader_ecg_img, num_epochs-1, method4, results_ecg, fold, n)
                
            ### calculate the running time of the model
            run_time_vgg_ecg = datetime.now() - start_vgg_ecg
            print('the training time of method {} is {}'.format(method4, run_time_vgg_ecg))
            ### get the size of the model
            p = pickle.dumps(vgg_ecg)
            vgg_ecg_size = sys.getsizeof(p)
            print("the size of method {} is {}".format(method4, vgg_ecg_size))
            

                
            ### create train set and test set for 1d cnn
            train_loader_ecg = torch.utils.data.DataLoader(
                              dataset_ecg, batch_size=batch_size_ecg, sampler=train_subsampler)
            
            testset_ecg2 = torch.utils.data.Subset(dataset_ecg, test_ids.tolist())
            test_loader_ecg = torch.utils.data.DataLoader(testset_ecg2, batch_size=batch_size_ecg, shuffle=False)             
            
            
            method5 = '1d_cnn'
            cnn_ecg=CNN_1d().cuda()
            cnn_ecg.apply(reset_weights)
    
            num_epochs = 1500
            start_cnn_ecg = datetime.now() 
            train_model(cnn_ecg, train_loader_ecg, test_loader_ecg, num_epochs, results_ecg, method5, fold, n)
            y_cnn_e, pred_cnn_e, acc_cnn_e, rest_pre_cnn_e, focus_pre_cnn_e, rest_rec_cnn_e, focus_rec_cnn_e =\
                predict_img(cnn_ecg, test_loader_ecg, num_epochs-1, method5, results_ecg, fold, n)
            
            ### calculate the running time of the model
            run_time_cnn_ecg = datetime.now() - start_cnn_ecg
            print('the training time of method {} is {}'.format(method5, run_time_cnn_ecg))
            ### get the size of the model
            p = pickle.dumps(cnn_ecg)
            cnn_ecg_size = sys.getsizeof(p)
            print("the size of method {} is {}".format(method5, cnn_ecg_size))
            

                
            ### create train set and test set for svm and xgboost
            x_train_ecg, y_train_ecg = x_ecg[train_ids], y_ecg[train_ids]
            x_test_ecg, y_test_ecg = x_ecg[test_ids], y_ecg[test_ids]           
            x_train_ecg, y_train_ecg = shuffle(x_train_ecg, y_train_ecg) ## only shuffle train dataset 
            
            method6 = 'svm_ecg'
            svm_ecg = svm.SVC(kernel='rbf', probability=True)
            start_svm_ecg = datetime.now() 
            svm_ecg.fit(x_train_ecg, y_train_ecg)
            pred_svm_e, acc_svm_e, rest_pre_svm_e, focus_pre_svm_e, rest_rec_svm_e, focus_rec_svm_e = \
                predict_svm_xgb(svm_ecg, x_test_ecg, y_test_ecg, results_ecg, method6, fold, n)               
            
            ### calculate the running time of the model
            run_time_svm_ecg= datetime.now() - start_svm_ecg
            print('the training time of method {} is {}'.format(method6, run_time_svm_ecg))   
            ### get the size of the model
            p = pickle.dumps(svm_ecg)
            svm_ecg_size = sys.getsizeof(p)
            print("the size of method {} is {}".format(method6,svm_ecg_size))
            
            
            method7 = 'xgb_ecg'
            xgb_ecg= xgb.XGBClassifier(learning_rate =0.1, n_estimators=200, max_depth=5, min_child_weight=1, gamma=0,  subsample=0.8,\
                                  colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)  
                
            pred_xgb_e, acc_xgb_e, rest_pre_xgb_e, focus_pre_xgb_e, rest_rec_xgb_e, focus_rec_xgb_e = \
                predict_svm_xgb(xgb_ecg, x_test_ecg, y_test_ecg, results_ecg, method7, fold, n)                
                
            start_xgb_ecg = datetime.now()    
            xgb_ecg.fit(x_train_ecg,y_train_ecg)
            ### calculate the running time of the model
            run_time_xgb_ecg= datetime.now() - start_xgb_ecg
            print('the training time of method {} is {}'.format(method7, run_time_xgb_ecg))   
            ### get the size of the model
            p = pickle.dumps(xgb_ecg)
            xgb_ecg_size = sys.getsizeof(p)
            print("the size of method {} is {}".format(method7,xgb_ecg_size))            
            

            
            print('y_test_ecg == y_vgg_e ?', y_test_ecg==y_vgg_e)  
            print('y_test_ecg == y_cnn_e',y_test_ecg == y_cnn_e)
            # print('y_test_224 == y_test_ecg',y_test_224== y_test_ecg)
            
            y_ensem_ecg = []
            for i in range(len(y_test_ecg)):
                pred_ave_ensem_e = (pred_vgg_e[i] + pred_cnn_e[i] + pred_svm_e[i] + pred_xgb_e[i])/4
                
                if pred_ave_ensem_e >= 0.5:
                    y_ensem_ecg.append(1)
                else:
                    y_ensem_ecg.append(0)
                
            
            ## calculate how many samples are predicted correctly.
            for t, e in zip(y_test_ecg, y_ensem_ecg):
                if t == e and t.item() == 0:
                    rest_true_ecg += 1
                elif t != e and t.item() == 0:
                    focus_false_ecg += 1
                elif t == e and t.item() == 1:
                    focus_true_ecg += 1
                else:
                    rest_false_ecg += 1
        
            rest_pre_ensem_e = rest_true_ecg/(rest_true_ecg+rest_false_ecg)
            focus_pre_ensem_e = focus_true_ecg/(focus_true_ecg+focus_false_ecg)
            
            rest_rec_ensem_e = rest_true_ecg/(rest_true_ecg+focus_false_ecg)
            focus_rec_ensem_e = focus_true_ecg/(focus_true_ecg+rest_false_ecg)
            
            acc_ensem_e = accuracy_score(y_test_ecg, y_ensem_ecg)

    
            rest_pre_ensem_ecg.append(rest_pre_ensem_e)
            focus_pre_ensem_ecg.append(focus_pre_ensem_e)
            rest_pre_vgg_ecg.append(rest_pre_vgg_e)
            focus_pre_vgg_ecg.append(focus_pre_vgg_e)
            rest_pre_cnn_ecg.append(rest_pre_cnn_e)
            focus_pre_cnn_ecg.append(focus_pre_cnn_e)
            rest_pre_svm_ecg.append(rest_pre_svm_e)
            focus_pre_svm_ecg.append(focus_pre_svm_e)            
            rest_pre_xgb_ecg.append(rest_pre_xgb_e)
            focus_pre_xgb_ecg.append(focus_pre_xgb_e)  
            
            rest_rec_ensem_ecg.append(rest_rec_ensem_e)
            focus_rec_ensem_ecg.append(focus_rec_ensem_e)
            rest_rec_vgg_ecg.append(rest_rec_vgg_e)
            focus_rec_vgg_ecg.append(focus_rec_vgg_e)
            rest_rec_cnn_ecg.append(rest_rec_cnn_e)
            focus_rec_cnn_ecg.append(focus_rec_cnn_e)
            rest_rec_svm_ecg.append(rest_rec_svm_e)
            focus_rec_svm_ecg.append(focus_rec_svm_e)            
            rest_rec_xgb_ecg.append(rest_rec_xgb_e)
            focus_rec_xgb_ecg.append(focus_rec_xgb_e)  
            
            acc_ensem_ecg.append(acc_ensem_e)
            acc_vgg_ecg.append(acc_vgg_e)
            acc_cnn_ecg.append(acc_cnn_e)
            acc_svm_ecg.append(acc_svm_e)
            acc_xgb_ecg.append(acc_xgb_e)
                                           
            fig_path = results_ecg + "/emsemble_figures/"
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            plot_confusion2(y_vgg_e, y_ensem_ecg, args.method, fig_path, fold, n, labels = [0,1])  
    
    

    
    ###########################################################################
    ### results of ecg signal based methods
    ### results of accuracy
    acc_ensem_mean_ecg = np.array(acc_ensem_ecg).mean()
    acc_ensem_std_ecg = np.array(acc_ensem_ecg).std()
    acc_ensem_conf_ecg = st.t.interval(0.95, len(acc_ensem_ecg)-1, loc=np.mean(acc_ensem_ecg), scale=st.sem(acc_ensem_ecg))
    acc_vgg_mean_ecg = np.array(acc_vgg_ecg).mean()
    acc_vgg_std_ecg = np.array(acc_vgg_ecg).std()
    acc_vgg_conf_ecg = st.t.interval(0.95, len(acc_vgg_ecg)-1, loc=np.mean(acc_vgg_ecg), scale=st.sem(acc_vgg_ecg))
    acc_cnn_mean_ecg = np.array(acc_cnn_ecg).mean()
    acc_cnn_std_ecg = np.array(acc_cnn_ecg).std()
    acc_cnn_conf_ecg = st.t.interval(0.95, len(acc_cnn_ecg)-1, loc=np.mean(acc_cnn_ecg), scale=st.sem(acc_cnn_ecg))
    acc_svm_mean_ecg = np.array(acc_svm_ecg).mean()
    acc_svm_std_ecg = np.array(acc_svm_ecg).std()
    acc_svm_conf_ecg = st.t.interval(0.95, len(acc_svm_ecg)-1, loc=np.mean(acc_svm_ecg), scale=st.sem(acc_svm_ecg))
    acc_xgb_mean_ecg = np.array(acc_xgb_ecg).mean()
    acc_xgb_std_ecg = np.array(acc_xgb_ecg).std()
    acc_xgb_conf_ecg = st.t.interval(0.95, len(acc_xgb_ecg)-1, loc=np.mean(acc_xgb_ecg), scale=st.sem(acc_xgb_ecg))

    ### results of precision
    rest_pre_ensem_mean_ecg = np.array(rest_pre_ensem_ecg).mean()
    rest_pre_ensem_std_ecg = np.array(rest_pre_ensem_ecg).std() 
    rest_pre_ensem_conf_ecg = st.t.interval(0.95, len(rest_pre_ensem_ecg)-1, loc=np.mean(rest_pre_ensem_ecg), scale=st.sem(rest_pre_ensem_ecg))
    focus_pre_ensem_mean_ecg = np.array(focus_pre_ensem_ecg).mean()
    focus_pre_ensem_std_ecg = np.array(focus_pre_ensem_ecg).std()
    focus_pre_ensem_conf_ecg = st.t.interval(0.95, len(focus_pre_ensem_ecg)-1, loc=np.mean(focus_pre_ensem_ecg), scale=st.sem(focus_pre_ensem_ecg))
    
    rest_pre_vgg_mean_ecg = np.array(rest_pre_vgg_ecg).mean()
    rest_pre_vgg_std_ecg = np.array(rest_pre_vgg_ecg).std() 
    rest_pre_vgg_conf_ecg = st.t.interval(0.95, len(rest_pre_vgg_ecg)-1, loc=np.mean(rest_pre_vgg_ecg), scale=st.sem(rest_pre_vgg_ecg))
    focus_pre_vgg_mean_ecg = np.array(focus_pre_vgg_ecg).mean()
    focus_pre_vgg_std_ecg = np.array(focus_pre_vgg_ecg).std()
    focus_pre_vgg_conf_ecg = st.t.interval(0.95, len(focus_pre_vgg_ecg)-1, loc=np.mean(focus_pre_vgg_ecg), scale=st.sem(focus_pre_vgg_ecg))

    rest_pre_cnn_mean_ecg = np.array(rest_pre_cnn_ecg).mean()
    rest_pre_cnn_std_ecg = np.array(rest_pre_cnn_ecg).std() 
    rest_pre_cnn_conf_ecg = st.t.interval(0.95, len(rest_pre_cnn_ecg)-1, loc=np.mean(rest_pre_cnn_ecg), scale=st.sem(rest_pre_cnn_ecg))
    focus_pre_cnn_mean_ecg = np.array(focus_pre_cnn_ecg).mean()
    focus_pre_cnn_std_ecg = np.array(focus_pre_cnn_ecg).std()
    focus_pre_cnn_conf_ecg = st.t.interval(0.95, len(focus_pre_cnn_ecg)-1, loc=np.mean(focus_pre_cnn_ecg), scale=st.sem(focus_pre_cnn_ecg))
    
    rest_pre_svm_mean_ecg = np.array(rest_pre_svm_ecg).mean()
    rest_pre_svm_std_ecg = np.array(rest_pre_svm_ecg).std()  
    rest_pre_svm_conf_ecg = st.t.interval(0.95, len(rest_pre_svm_ecg)-1, loc=np.mean(rest_pre_svm_ecg), scale=st.sem(rest_pre_svm_ecg))
    focus_pre_svm_mean_ecg = np.array(focus_pre_svm_ecg).mean()
    focus_pre_svm_std_ecg = np.array(focus_pre_svm_ecg).std() 
    focus_pre_svm_conf_ecg = st.t.interval(0.95, len(focus_pre_svm_ecg)-1, loc=np.mean(focus_pre_svm_ecg), scale=st.sem(focus_pre_svm_ecg))
    
    rest_pre_xgb_mean_ecg = np.array(rest_pre_xgb_ecg).mean()
    rest_pre_xgb_std_ecg = np.array(rest_pre_xgb_ecg).std() 
    rest_pre_xgb_conf_ecg = st.t.interval(0.95, len(rest_pre_xgb_ecg)-1, loc=np.mean(rest_pre_xgb_ecg), scale=st.sem(rest_pre_xgb_ecg))
    focus_pre_xgb_mean_ecg = np.array(focus_pre_xgb_ecg).mean()
    focus_pre_xgb_std_ecg = np.array(focus_pre_xgb_ecg).std()
    focus_pre_xgb_conf_ecg = st.t.interval(0.95, len(focus_pre_xgb_ecg)-1, loc=np.mean(focus_pre_xgb_ecg), scale=st.sem(focus_pre_xgb_ecg))
    
    ### results of recall
    rest_rec_ensem_mean_ecg = np.array(rest_rec_ensem_ecg).mean()
    rest_rec_ensem_std_ecg = np.array(rest_rec_ensem_ecg).std() 
    rest_rec_ensem_conf_ecg = st.t.interval(0.95, len(rest_rec_ensem_ecg)-1, loc=np.mean(rest_rec_ensem_ecg), scale=st.sem(rest_rec_ensem_ecg))
    focus_rec_ensem_mean_ecg = np.array(focus_rec_ensem_ecg).mean()
    focus_rec_ensem_std_ecg = np.array(focus_rec_ensem_ecg).std()
    focus_rec_ensem_conf_ecg = st.t.interval(0.95, len(focus_rec_ensem_ecg)-1, loc=np.mean(focus_rec_ensem_ecg), scale=st.sem(focus_rec_ensem_ecg))
    
    rest_rec_vgg_mean_ecg = np.array(rest_rec_vgg_ecg).mean()
    rest_rec_vgg_std_ecg = np.array(rest_rec_vgg_ecg).std() 
    rest_rec_vgg_conf_ecg = st.t.interval(0.95, len(rest_rec_vgg_ecg)-1, loc=np.mean(rest_rec_vgg_ecg), scale=st.sem(rest_rec_vgg_ecg))
    focus_rec_vgg_mean_ecg = np.array(focus_rec_vgg_ecg).mean()
    focus_rec_vgg_std_ecg = np.array(focus_rec_vgg_ecg).std()
    focus_rec_vgg_conf_ecg = st.t.interval(0.95, len(focus_rec_vgg_ecg)-1, loc=np.mean(focus_rec_vgg_ecg), scale=st.sem(focus_rec_vgg_ecg))

    rest_rec_cnn_mean_ecg = np.array(rest_rec_cnn_ecg).mean()
    rest_rec_cnn_std_ecg = np.array(rest_rec_cnn_ecg).std() 
    rest_rec_cnn_conf_ecg = st.t.interval(0.95, len(rest_rec_cnn_ecg)-1, loc=np.mean(rest_rec_cnn_ecg), scale=st.sem(rest_rec_cnn_ecg))
    focus_rec_cnn_mean_ecg = np.array(focus_rec_cnn_ecg).mean()
    focus_rec_cnn_std_ecg = np.array(focus_rec_cnn_ecg).std()
    focus_rec_cnn_conf_ecg = st.t.interval(0.95, len(focus_rec_cnn_ecg)-1, loc=np.mean(focus_rec_cnn_ecg), scale=st.sem(focus_rec_cnn_ecg))
    
    rest_rec_svm_mean_ecg = np.array(rest_rec_svm_ecg).mean()
    rest_rec_svm_std_ecg = np.array(rest_rec_svm_ecg).std()  
    rest_rec_svm_conf_ecg = st.t.interval(0.95, len(rest_rec_svm_ecg)-1, loc=np.mean(rest_rec_svm_ecg), scale=st.sem(rest_rec_svm_ecg))
    focus_rec_svm_mean_ecg = np.array(focus_rec_svm_ecg).mean()
    focus_rec_svm_std_ecg = np.array(focus_rec_svm_ecg).std() 
    focus_rec_svm_conf_ecg = st.t.interval(0.95, len(focus_rec_svm_ecg)-1, loc=np.mean(focus_rec_svm_ecg), scale=st.sem(focus_rec_svm_ecg))
    
    rest_rec_xgb_mean_ecg = np.array(rest_rec_xgb_ecg).mean()
    rest_rec_xgb_std_ecg = np.array(rest_rec_xgb_ecg).std() 
    rest_rec_xgb_conf_ecg = st.t.interval(0.95, len(rest_rec_xgb_ecg)-1, loc=np.mean(rest_rec_xgb_ecg), scale=st.sem(rest_rec_xgb_ecg))
    focus_rec_xgb_mean_ecg = np.array(focus_rec_xgb_ecg).mean()
    focus_rec_xgb_std_ecg = np.array(focus_rec_xgb_ecg).std()
    focus_rec_xgb_conf_ecg = st.t.interval(0.95, len(focus_rec_xgb_ecg)-1, loc=np.mean(focus_rec_xgb_ecg), scale=st.sem(focus_rec_xgb_ecg))    
    
    
    ### print accuracy
    print("ecg signal method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, acc_ensem_mean_ecg, acc_ensem_std_ecg,acc_ensem_conf_ecg[0],acc_ensem_conf_ecg[1]))
    print("ecg signal method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method4, acc_vgg_mean_ecg, acc_vgg_std_ecg, acc_vgg_conf_ecg[0],acc_vgg_conf_ecg[1]))
    print("ecg signal method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method5, acc_cnn_mean_ecg, acc_cnn_std_ecg, acc_cnn_conf_ecg[0],acc_cnn_conf_ecg[1]))
    print("ecg signal method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method6, acc_svm_mean_ecg, acc_svm_std_ecg, acc_svm_conf_ecg[0],acc_svm_conf_ecg[1]))
    print("ecg signal method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method7, acc_xgb_mean_ecg, acc_xgb_std_ecg, acc_xgb_conf_ecg[0],acc_xgb_conf_ecg[1]))
    
    ### print precision
    print("ecg signal method %s: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_pre_ensem_mean_ecg, rest_pre_ensem_std_ecg, rest_pre_ensem_conf_ecg[0], rest_pre_ensem_conf_ecg[1]))
    print("ecg signal method %s: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_pre_ensem_mean_ecg, focus_pre_ensem_std_ecg, focus_pre_ensem_conf_ecg[0],focus_pre_ensem_conf_ecg[1]))
    print("ecg signal method vgg: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_vgg_mean_ecg, rest_pre_vgg_std_ecg, rest_pre_vgg_conf_ecg[0],rest_pre_vgg_conf_ecg[1]))
    print("ecg signal method vgg: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_vgg_mean_ecg, focus_pre_vgg_std_ecg, focus_pre_vgg_conf_ecg[0], focus_pre_vgg_conf_ecg[1]))
    print("ecg signal method cnn: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_cnn_mean_ecg, rest_pre_cnn_std_ecg, rest_pre_cnn_conf_ecg[0],rest_pre_cnn_conf_ecg[1]))
    print("ecg signal method cnn: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_cnn_mean_ecg, focus_pre_cnn_std_ecg, focus_pre_cnn_conf_ecg[0], focus_pre_cnn_conf_ecg[1]))
    print("ecg signal method svm: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_svm_mean_ecg, rest_pre_svm_std_ecg, rest_pre_svm_conf_ecg[0], rest_pre_svm_conf_ecg[1]))
    print("ecg signal method svm: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_svm_mean_ecg, focus_pre_svm_std_ecg, focus_pre_svm_conf_ecg[0], focus_pre_svm_conf_ecg[1]))   
    print("ecg signal method xgb: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_xgb_mean_ecg, rest_pre_xgb_std_ecg, rest_pre_xgb_conf_ecg[0], rest_pre_xgb_conf_ecg[1]))
    print("ecg signal method xgb: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_xgb_mean_ecg, focus_pre_xgb_std_ecg, focus_pre_xgb_conf_ecg[0], focus_pre_xgb_conf_ecg[1]))

    ### print recall
    print("ecg signal method %s: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_rec_ensem_mean_ecg, rest_rec_ensem_std_ecg, rest_rec_ensem_conf_ecg[0], rest_rec_ensem_conf_ecg[1]))
    print("ecg signal method %s: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_rec_ensem_mean_ecg, focus_rec_ensem_std_ecg, focus_rec_ensem_conf_ecg[0],focus_rec_ensem_conf_ecg[1]))
    print("ecg signal method vgg: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_vgg_mean_ecg, rest_rec_vgg_std_ecg, rest_rec_vgg_conf_ecg[0],rest_rec_vgg_conf_ecg[1]))
    print("ecg signal method vgg: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_vgg_mean_ecg, focus_rec_vgg_std_ecg, focus_rec_vgg_conf_ecg[0], focus_rec_vgg_conf_ecg[1]))
    print("ecg signal method cnn: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_cnn_mean_ecg, rest_rec_cnn_std_ecg, rest_rec_cnn_conf_ecg[0],rest_rec_cnn_conf_ecg[1]))
    print("ecg signal method cnn: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_cnn_mean_ecg, focus_rec_cnn_std_ecg, focus_rec_cnn_conf_ecg[0], focus_rec_cnn_conf_ecg[1]))
    print("ecg signal method svm: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_svm_mean_ecg, rest_rec_svm_std_ecg, rest_rec_svm_conf_ecg[0], rest_rec_svm_conf_ecg[1]))
    print("ecg signal method svm: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_svm_mean_ecg, focus_rec_svm_std_ecg, focus_rec_svm_conf_ecg[0], focus_rec_svm_conf_ecg[1]))   
    print("ecg signal method xgb: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_xgb_mean_ecg, rest_rec_xgb_std_ecg, rest_rec_xgb_conf_ecg[0], rest_rec_xgb_conf_ecg[1]))
    print("ecg signal method xgb: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_xgb_mean_ecg, focus_rec_xgb_std_ecg, focus_rec_xgb_conf_ecg[0], focus_rec_xgb_conf_ecg[1]))    

    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    results_f = results_ecg + '{}_restults.txt'.format(args.method)
    with open(results_f, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        ### write accuracy
        f.write("ecg signal method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, acc_ensem_mean_ecg, acc_ensem_std_ecg,acc_ensem_conf_ecg[0],acc_ensem_conf_ecg[1]))
        f.write("ecg signal method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method4, acc_vgg_mean_ecg, acc_vgg_std_ecg, acc_vgg_conf_ecg[0],acc_vgg_conf_ecg[1]))
        f.write("ecg signal method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method5, acc_cnn_mean_ecg, acc_cnn_std_ecg, acc_cnn_conf_ecg[0],acc_cnn_conf_ecg[1]))
        f.write("ecg signal method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method6, acc_svm_mean_ecg, acc_svm_std_ecg, acc_svm_conf_ecg[0],acc_svm_conf_ecg[1]))
        f.write("ecg signal method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method7, acc_xgb_mean_ecg, acc_xgb_std_ecg, acc_xgb_conf_ecg[0],acc_xgb_conf_ecg[1]))
        f.write('--'*40 +'\n')
        ### write precision
        f.write("ecg signal method %s: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_pre_ensem_mean_ecg, rest_pre_ensem_std_ecg, rest_pre_ensem_conf_ecg[0], rest_pre_ensem_conf_ecg[1]))
        f.write("ecg signal method %s: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_pre_ensem_mean_ecg, focus_pre_ensem_std_ecg, focus_pre_ensem_conf_ecg[0],focus_pre_ensem_conf_ecg[1]))
        f.write("ecg signal method vgg: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_vgg_mean_ecg, rest_pre_vgg_std_ecg, rest_pre_vgg_conf_ecg[0],rest_pre_vgg_conf_ecg[1]))
        f.write("ecg signal method vgg: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_vgg_mean_ecg, focus_pre_vgg_std_ecg, focus_pre_vgg_conf_ecg[0], focus_pre_vgg_conf_ecg[1]))
        f.write("ecg signal method cnn: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_cnn_mean_ecg, rest_pre_cnn_std_ecg, rest_pre_cnn_conf_ecg[0],rest_pre_cnn_conf_ecg[1]))
        f.write("ecg signal method cnn: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_cnn_mean_ecg, focus_pre_cnn_std_ecg, focus_pre_cnn_conf_ecg[0], focus_pre_cnn_conf_ecg[1]))
        f.write("ecg signal method svm: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_svm_mean_ecg, rest_pre_svm_std_ecg, rest_pre_svm_conf_ecg[0], rest_pre_svm_conf_ecg[1]))
        f.write("ecg signal method svm: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_svm_mean_ecg, focus_pre_svm_std_ecg, focus_pre_svm_conf_ecg[0], focus_pre_svm_conf_ecg[1]))   
        f.write("ecg signal method xgb: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_xgb_mean_ecg, rest_pre_xgb_std_ecg, rest_pre_xgb_conf_ecg[0], rest_pre_xgb_conf_ecg[1]))
        f.write("ecg signal method xgb: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_xgb_mean_ecg, focus_pre_xgb_std_ecg, focus_pre_xgb_conf_ecg[0], focus_pre_xgb_conf_ecg[1]))
        f.write('--'*40 +'\n')
        ### write recall
        f.write("ecg signal method %s: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_rec_ensem_mean_ecg, rest_rec_ensem_std_ecg, rest_rec_ensem_conf_ecg[0], rest_rec_ensem_conf_ecg[1]))
        f.write("ecg signal method %s: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_rec_ensem_mean_ecg, focus_rec_ensem_std_ecg, focus_rec_ensem_conf_ecg[0],focus_rec_ensem_conf_ecg[1]))
        f.write("ecg signal method vgg: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_vgg_mean_ecg, rest_rec_vgg_std_ecg, rest_rec_vgg_conf_ecg[0],rest_rec_vgg_conf_ecg[1]))
        f.write("ecg signal method vgg: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_vgg_mean_ecg, focus_rec_vgg_std_ecg, focus_rec_vgg_conf_ecg[0], focus_rec_vgg_conf_ecg[1]))
        f.write("ecg signal method cnn: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_cnn_mean_ecg, rest_rec_cnn_std_ecg, rest_rec_cnn_conf_ecg[0],rest_rec_cnn_conf_ecg[1]))
        f.write("ecg signal method cnn: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_cnn_mean_ecg, focus_rec_cnn_std_ecg, focus_rec_cnn_conf_ecg[0], focus_rec_cnn_conf_ecg[1]))
        f.write("ecg signal method svm: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_svm_mean_ecg, rest_rec_svm_std_ecg, rest_rec_svm_conf_ecg[0], rest_rec_svm_conf_ecg[1]))
        f.write("ecg signal method svm: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_svm_mean_ecg, focus_rec_svm_std_ecg, focus_rec_svm_conf_ecg[0], focus_rec_svm_conf_ecg[1]))   
        f.write("ecg signal method xgb: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_xgb_mean_ecg, rest_rec_xgb_std_ecg, rest_rec_xgb_conf_ecg[0], rest_rec_xgb_conf_ecg[1]))
        f.write("ecg signal method xgb: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_xgb_mean_ecg, focus_rec_xgb_std_ecg, focus_rec_xgb_conf_ecg[0], focus_rec_xgb_conf_ecg[1]))    

        f.write('--'*40 +'\n')
        f.write('ecg signal based: the size of method vgg is {}, and the runing time is {} \n'.format(vgg_ecg_size, run_time_vgg_ecg))
        f.write('ecg signal based: the size of method vgg is {}, and the runing time is {} \n'.format(cnn_ecg_size, run_time_cnn_ecg))
        f.write('ecg signal based: the size of method svm is {}, and the runing time is {} \n'.format(svm_ecg_size, run_time_svm_ecg))
        f.write('ecg signal based: the size of method xgb is {}, and the runing time is {} \n'.format(xgb_ecg_size, run_time_xgb_ecg))
    
    ###########################################################################
    ### resutls of ensemble on ensemble
    
                
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

    
