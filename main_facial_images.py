#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 12:17:56 2021

@author: xiaoyan
"""

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
from scipy.optimize import differential_evolution
from numpy import tensordot
from numpy.linalg import norm


    
def predict_img(args, model, test_loader, epoch, method, path, fold, n, test_ids): 
    results_path = path + '{}/'.format(method)
    if not os.path.exists(results_path):
            os.makedirs(results_path)  
    if args.test:
        checkpoint = torch.load(results_path + 'repeat{}_fold{}_epoch{}.pt'.format(n, fold, epoch))
        model.load_state_dict(checkpoint["model_state_dict"]) 
            
    results_f = results_path + '{}_restults.txt'.format(method) 
    
    if args.cuda:    
        model.cuda()
    model.eval()
    test_acc = 0
    total_num = 0
    rest_true = 0
    rest_false = 0
    focus_true = 0
    focus_false = 0
    correct_id_res = []
    incorrect_id_res = []
    correct_id_foc = []
    incorrect_id_foc = []
    
    y_pred = []
    y_true = []
    y_prob = []
    
    i = 0
    for x_test, test_y in test_loader:
        total_num += len(test_y)
        with torch.no_grad():
            yhats, _ = model(to_gpu(args.cuda, x_test))
        
        yhats = yhats.cpu().detach().numpy()       
        ### the output probability        
        prob = [[1-y.item(),y.item()] for y in yhats]
        y_prob.extend(prob)
                
        ### if preds > 0.5, preds = 1, otherwise, preds = 0
        pred = [y.item() > 0.5 for y in yhats]
        y_pred.extend(list(map(int, pred)))
        
        test_y = test_y.detach().numpy()    
        y_true.extend(test_y.tolist())
        
        test_acc += sum(test_y == np.array(pred))
        
       
        ## calculate how many samples are predicted correctly.
        
     
        for t, p in zip(test_y, pred):
            if t == p and t.item() == 0:
                rest_true += 1
                correct_id_res.append(test_ids[i])
            elif t != p and t.item() == 0:
                focus_false += 1
                incorrect_id_res.append(test_ids[i])
            elif t == p and t.item() == 1:
                focus_true += 1
                correct_id_foc.append(test_ids[i])
            else:
                rest_false += 1
                incorrect_id_foc.append(test_ids[i])
            i+=1

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
    
    ### save correctly classified sample idex into files
    pred_pkl = results_path + '/pred_pkl/'
    if not os.path.exists(pred_pkl):
        os.makedirs(pred_pkl) 
        
    pred_idx1 = pred_pkl + '{}_repeat{}_fold{}_pred_index1.pkl'.format(method, n, fold)
    pred_idx2 = pred_pkl + '{}_repeat{}_fold{}_pred_index2.pkl'.format(method, n, fold)
    true_ind = pred_pkl + '{}_repeat{}_fold{}_true_index.pkl'.format(method, n, fold)
    
    pred_arry = np.array([correct_id_res, incorrect_id_res, correct_id_foc, incorrect_id_foc])
    pickle.dump(pred_arry, open(pred_idx1, "wb"))

    pred_res = []
    pred_foc = []
    pred_res.extend(correct_id_res)
    pred_res.extend(incorrect_id_foc)
    pred_foc.extend(correct_id_foc)
    pred_foc.extend(incorrect_id_res)
    pred_res_foc = np.array([pred_res, pred_foc])
    pickle.dump(pred_res_foc, open(pred_idx2, "wb"))  

    true_res = []
    true_foc = []
    true_res.extend(correct_id_res)
    true_res.extend(incorrect_id_res)
    true_foc.extend(correct_id_foc)
    true_foc.extend(incorrect_id_foc)    
    
    true_res_foc = np.array([true_res, true_foc])
    pickle.dump(true_res_foc, open(true_ind, "wb")) 
    
    ### save the predicted probility and class label into files
    results_prob = pred_pkl + '{}_repeat{}_fold{}_probs.pkl'.format(method, n, fold)
    results_pred = pred_pkl + '{}_repeat{}_fold{}_preds.pkl'.format(method, n, fold)
    results_true =  pred_pkl + '{}_repeat{}_fold{}_true.pkl'.format(method, n, fold)  
    
    pickle.dump(y_prob, open(results_prob, "wb"))
    pickle.dump(y_pred, open(results_pred, "wb"))
    pickle.dump(y_true, open(results_true, "wb"))
           
    fig_path = results_path + "/{}_figures/".format(method)  
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)            
                
    plot_confusion2(y_true, y_pred, method, fig_path, fold, n, labels = [0,1])
    
    return  y_true, y_pred, y_prob, acc, rest_precision, focus_precision, rest_recall, focus_recall, correct_id_res, incorrect_id_res, correct_id_foc, incorrect_id_foc

def predict_svm_xgb(args, model, x_test, y_test, path, method, fold, n, test_ids):
    rest_true = 0
    rest_false = 0
    focus_true = 0
    focus_false = 0  
    focus_pre_ls = []
    rest_pre_ls = []
    acc_ls = []
    
    correct_id_res = []
    incorrect_id_res = []
    correct_id_foc = []
    incorrect_id_foc = []
       
    results_path = path + '{}/'.format(method)   
    if not os.path.exists(results_path):
            os.makedirs(results_path) 
     
    pk_f = results_path +'repeat{}_fold{}.pkl'.format(n, fold)
    if args.test:
        with open(pk_f, 'rb') as f:
            model = pickle.load(f)
        
    results_f = results_path + '{}_restults.txt'.format(method)
    
    yhats = model.predict_proba(x_test)
    # y_prob = yhats[:,1]
    y_pred = np.argmax(yhats, axis=1).tolist()
    print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")
  
    ## calculate how many samples are predicted correctly.
    
    for t, p, i in zip(y_test, y_pred, test_ids):
        if t == p and t.item() == 0:
            rest_true += 1
            correct_id_res.append(i)
        elif t != p and t.item() == 0:
            focus_false += 1
            incorrect_id_res.append(i)
        elif t == p and t.item() == 1:
            focus_true += 1
            correct_id_foc.append(i)
        else:
            rest_false += 1
            incorrect_id_foc.append(i)
        
            
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
    
    ### save correctly classified sample idex into files
    pred_pkl = results_path + '/pred_pkl/'
    if not os.path.exists(pred_pkl):
        os.makedirs(pred_pkl) 
    pred_idx1 = pred_pkl + '{}_repeat{}_fold{}_pred_index1.pkl'.format(method, n, fold)
    pred_idx2 = pred_pkl + '{}_repeat{}_fold{}_pred_index2.pkl'.format(method, n, fold)
    true_ind = pred_pkl + '{}_repeat{}_fold{}_true_index.pkl'.format(method, n, fold)
    
    pred_arry = np.array([correct_id_res, incorrect_id_res, correct_id_foc, incorrect_id_foc])
    pickle.dump(pred_arry, open(pred_idx1, "wb"))

    pred_res = []
    pred_foc = []
    pred_res.extend(correct_id_res)
    pred_res.extend(incorrect_id_foc)
    pred_foc.extend(correct_id_foc)
    pred_foc.extend(incorrect_id_res)
    pred_res_foc = np.array([pred_res, pred_foc])
    pickle.dump(pred_res_foc, open(pred_idx2, "wb"))  

    true_res = []
    true_foc = []
    true_res.extend(correct_id_res)
    true_res.extend(incorrect_id_res)
    true_foc.extend(correct_id_foc)
    true_foc.extend(incorrect_id_foc)    
    
    true_res_foc = np.array([true_res, true_foc])
    pickle.dump(true_res_foc, open(true_ind, "wb")) 
    
    ### save the predicted probility and class label into files
    results_prob = pred_pkl + '{}_repeat{}_fold{}_probs.pkl'.format(method, n, fold)
    results_pred = pred_pkl + '{}_repeat{}_fold{}_preds.pkl'.format(method, n, fold)
    results_true =  pred_pkl + '{}_repeat{}_fold{}_true.pkl'.format(method, n, fold)  
    
    pickle.dump(yhats, open(results_prob, "wb"))
    pickle.dump(y_pred, open(results_pred, "wb"))
    pickle.dump(y_test, open(results_true, "wb"))
               
    
    return y_pred, yhats, acc, rest_precision, focus_precision, rest_recall, focus_recall, correct_id_res, incorrect_id_res, correct_id_foc, incorrect_id_foc

def train_model(model, train_loader, test_loader, num_epochs, path, method, fold, n):
    results_path = path + '{}/'.format(method) 
    if not os.path.exists(results_path):
            os.makedirs(results_path)     

    
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    criterion = nn.BCELoss()
    if args.cuda:
        criterion = criterion.cuda()

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
            preds, _ = model(to_gpu(args.cuda, x_train))
            optimizer.zero_grad()

            loss = criterion(preds.squeeze(), to_gpu(args.cuda, y_train).float())
            
            loss.backward()
            optimizer.step()
            
            losses += loss.item()
            
        avg_loss = losses / len(train_loader)            
        train_losses.append(loss.item()) 

        with torch.no_grad():
            for x_val, y_val in test_loader:
                x_val = to_gpu(args.cuda, x_val)
                y_val = to_gpu(args.cuda, y_val)
                
                model.eval()
    
                yhat, _ = model(x_val)
                val_l = criterion(yhat.squeeze(), y_val.float())
                val_loss += val_l.item()
                
            val_losses.append(val_loss)
            
        print('Repeat {} Fold {} Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(n, fold,
            epoch, num_epochs, i, len(train_loader), loss.item()))
        if epoch == num_epochs-1:
            checkpoint = results_path +'repeat{}_fold{}_epoch{}.pt'.format(n, fold, epoch+1)
            save_objs(model, epoch+1, avg_loss, optimizer, checkpoint)
  
    plot_loss(method, train_losses, val_losses, results_path)
    
    
def main(args,results_face):
    
    ###########################################################################
    ### create dataset for facial image
    image_path = './data/images_224x224/'
    
    image_dir = image_path + 'images/'
    img_csv = 'image.csv'
    
    # check_path_cnn = args.results + 'checkpt_cnn'
    # check_path_vgg = args.results + 'checkpt_vgg'
    
    k_folds = 5
    repeat = 3
    seed = [0, 42, 10]
       
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    _, _, dataset_224 = create_datasets(args.batch_size,transform, image_path, image_dir, img_csv)
     
    acc_ensem_fac0 = []
    acc_ensem_fac = []
    acc_vgg_fac = []
    acc_svm_fac = []
    acc_xgb_fac = []
    
    rest_pre_ensem_fac0 = []
    focus_pre_ensem_fac0 = []
    rest_pre_ensem_fac = []
    focus_pre_ensem_fac = []
    rest_pre_vgg_fac = []
    focus_pre_vgg_fac = []
    rest_pre_svm_fac = []
    focus_pre_svm_fac = []
    rest_pre_xgb_fac = []
    focus_pre_xgb_fac = []

    rest_rec_ensem_fac0 = []
    focus_rec_ensem_fac0 = []
    rest_rec_ensem_fac = []
    focus_rec_ensem_fac = []
    rest_rec_vgg_fac = []
    focus_rec_vgg_fac = []
    rest_rec_svm_fac = []
    focus_rec_svm_fac = []
    rest_rec_xgb_fac = []
    focus_rec_xgb_fac = []    
    
    
    
    x_224 = []
    y_224 = []    
    for i in range(len(dataset_224)):    
        x_224.append(dataset_224[i][0].numpy().flatten())
        y_224.append(dataset_224[i][1].numpy())
    x_224 = np.asarray(x_224)
    y_224 = np.asarray(y_224)
    
    ###########################################################################
    ### create dataset for ecg signals  
    ### ecg image dataset
    image_path_ecg = './data/ecg_img/'
    image_dir_ecg = image_path_ecg + 'images/'  
    batch_size_ecg = 256
    _, _, dataset_ecg_img = create_datasets(batch_size_ecg, transform, image_path_ecg, image_dir_ecg, img_csv)   
    
    for n in range(repeat):
        kfold = KFold(n_splits=k_folds, random_state= seed[n], shuffle=True)
        
        vgg_correct_res = []
        vgg_incorrect_res = []
        vgg_correct_foc = []
        vgg_incorrect_foc = []
        
        svm_correct_res = []
        svm_incorrect_res = []
        svm_correct_foc = []
        svm_incorrect_foc = []

        xgb_correct_res = []
        xgb_incorrect_res = []
        xgb_correct_foc = []
        xgb_incorrect_foc = []
        
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset_224)):
            print(f'FOLD {fold}')
            start = datetime.now() 
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

            method1 = 'Pretrained_VGG_face'
            vgg_face = alexnet()
            if args.cuda:
                vgg_face = vgg_face.cuda()
            reset_weights_vgg(vgg_face)
            num_epochs = 150
            
            start_vgg = datetime.now() 
            if not args.test:
                train_model(vgg_face, train_loader_224, test_loader_224, num_epochs, results_face, method1, fold, n)  

            
            ### calculate the training time of the model
            train_time = datetime.now()
            train_time_vgg = train_time - start_vgg
            print('the training time of method {} is {}'.format(method1, train_time_vgg))
            ### get the size of the model
            p = pickle.dumps(vgg_face)
            vgg_size = sys.getsizeof(p)
            print("the size of method {} is {}".format(method1,vgg_size ))
                
            y_vgg_f, pred_vgg_f, prob_vgg_f, acc_vgg_f, rest_pre_vgg_f, focus_pre_vgg_f, rest_rec_vgg_f, focus_rec_vgg_f,\
            correct_res_vgg, incorrect_res_vgg, correct_foc_vgg, incorrect_foc_vgg = predict_img(args, vgg_face, test_loader_224, num_epochs, method1, results_face, fold, n, test_ids)            
            test_time_vgg = datetime.now() - train_time
            print('the testing time of method {} is {}'.format(method1, test_time_vgg))
            
            method2 = 'SVM_face'
            svm_face= svm.SVC(kernel='poly', probability=True)
            path_svm_face = results_face + '{}/'.format(method2) 
            if not os.path.exists(path_svm_face):
                os.makedirs(path_svm_face) 
            pk_svm_face = path_svm_face + 'repeat{}_fold{}.pkl'.format(n, fold)
            # svm_face= svm.SVC(kernel='rbf', C=5.0, probability=True)
            start_svm = datetime.now() 
            if not args.test:
                svm_face.fit(x_train_224,y_train_224)
            ### save trained svm_face
                with open(pk_svm_face, 'wb') as f1:
                    pickle.dump(svm_face, f1)
          
            ### calculate the training time of the model
            train_time = datetime.now()
            train_time_svm= train_time - start_svm
            print('the training time of method {} is {}'.format(method2, train_time_svm))   
            ### get the size of the model
            p = pickle.dumps(svm_face)
            svm_size = sys.getsizeof(p)
            print("the size of method {} is {}".format(method2,svm_size))
                   
            pred_svm_f, prob_svm_f, acc_svm_f, rest_pre_svm_f, focus_pre_svm_f, rest_rec_svm_f, focus_rec_svm_f, correct_res_svm, incorrect_res_svm, correct_foc_svm, incorrect_foc_svm = \
                predict_svm_xgb(args, svm_face, x_test_224, y_test_224, results_face, method2, fold, n, test_ids)   
            test_time_svm= datetime.now() - train_time
            print('the testing time of method {} is {}'.format(method2, test_time_svm))
            
            method3 = 'XGBoost_face'
            xgb_face= xgb.XGBClassifier(learning_rate =0.1, n_estimators=100, max_depth=5, min_child_weight=1, gamma=0,  subsample=0.8,\
                                  colsample_bytree=0.8, objective= 'binary:logistic', scale_pos_weight=1, seed=27)  
                
            path_xgb_face = results_face + '{}/'.format(method3) 
            if not os.path.exists(path_xgb_face):
                os.makedirs(path_xgb_face) 
            pk_xgb_face = path_xgb_face + 'repeat{}_fold{}.pkl'.format(n, fold)
            
            start_xgb = datetime.now() 
            if not args.test:
                xgb_face.fit(x_train_224,y_train_224)
            
                ### save trained xgb_face
                with open(pk_xgb_face, 'wb') as f2:
                    pickle.dump(xgb_face, f2)

            ### calculate the training time of the model
            train_time = datetime.now()
            train_time_xgb= train_time - start_xgb
            print('the training time of method {} is {}'.format(method3, train_time_xgb))   
            ### get the size of the model
            p = pickle.dumps(xgb_face)
            xgb_size = sys.getsizeof(p)
            print("the size of method {} is {}".format(method3,xgb_size))            
                   
            pred_xgb_f, prob_xgb_f, acc_xgb_f, rest_pre_xgb_f, focus_pre_xgb_f, rest_rec_xgb_f, focus_rec_xgb_f, correct_res_xgb, incorrect_res_xgb, correct_foc_xgb, incorrect_foc_xgb = \
                predict_svm_xgb(args, xgb_face, x_test_224, y_test_224, results_face, method3, fold, n, test_ids)
            # print('y_test_224 == y_vgg ?', y_test_224==y_vgg_f)
            test_time_xgb= datetime.now() - train_time
            print('the testing time of method {} is {}'.format(method3, test_time_xgb))
            

            preds = [prob_vgg_f, prob_svm_f, prob_xgb_f]
            acc_ensem_f0,y_ensem_f0, rest_true_f0,focus_false_f0, focus_true_f0, rest_false_f0 = equal_wight_ensemble(preds, y_test_224)
            acc_ensem_f,y_ensem_f, rest_true_f,focus_false_f, focus_true_f, rest_false_f = optimal_wight_ensemble(preds, y_test_224)


            rest_pre_ensem_f0 = rest_true_f0/(rest_true_f0+rest_false_f0)
            focus_pre_ensem_f0 = focus_true_f0/(focus_true_f0+focus_false_f0)           
            rest_rec_ensem_f0 = rest_true_f0/(rest_true_f0+focus_false_f0)
            focus_rec_ensem_f0 = focus_true_f0/(focus_true_f0+rest_false_f0)
            rest_pre_ensem_f = rest_true_f/(rest_true_f+rest_false_f)
            focus_pre_ensem_f = focus_true_f/(focus_true_f+focus_false_f)           
            rest_rec_ensem_f = rest_true_f/(rest_true_f+focus_false_f)
            focus_rec_ensem_f = focus_true_f/(focus_true_f+rest_false_f)
                        
        
            rest_pre_ensem_fac0.append(rest_pre_ensem_f0)
            focus_pre_ensem_fac0.append(focus_pre_ensem_f0)
            rest_pre_ensem_fac.append(rest_pre_ensem_f)
            focus_pre_ensem_fac.append(focus_pre_ensem_f)            
            
            rest_pre_vgg_fac.append(rest_pre_vgg_f)
            focus_pre_vgg_fac.append(focus_pre_vgg_f)
            rest_pre_svm_fac.append(rest_pre_svm_f)
            focus_pre_svm_fac.append(focus_pre_svm_f)            
            rest_pre_xgb_fac.append(rest_pre_xgb_f)
            focus_pre_xgb_fac.append(focus_pre_xgb_f)  
            
            rest_rec_ensem_fac0.append(rest_rec_ensem_f0)
            focus_rec_ensem_fac0.append(focus_rec_ensem_f0)
            rest_rec_ensem_fac.append(rest_rec_ensem_f)
            focus_rec_ensem_fac.append(focus_rec_ensem_f)
            
            rest_rec_vgg_fac.append(rest_rec_vgg_f)
            focus_rec_vgg_fac.append(focus_rec_vgg_f)
            rest_rec_svm_fac.append(rest_rec_svm_f)
            focus_rec_svm_fac.append(focus_rec_svm_f)            
            rest_rec_xgb_fac.append(rest_rec_xgb_f)
            focus_rec_xgb_fac.append(focus_rec_xgb_f)  
            
            acc_ensem_fac0.append(acc_ensem_f0)
            acc_ensem_fac.append(acc_ensem_f)
            acc_vgg_fac.append(acc_vgg_f)
            acc_svm_fac.append(acc_svm_f)
            acc_xgb_fac.append(acc_xgb_f)
                                           
            fig_path = results_face + "/emsemble_figures/"
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            plot_confusion2(y_vgg_f, y_ensem_f, args.method, fig_path, fold, n, labels = [0,1])    

            end = datetime.now() - start   
            print("total running time is:", end)
            
            vgg_correct_res.extend(correct_res_vgg)
            vgg_incorrect_res.extend(incorrect_res_vgg)
            vgg_correct_foc.extend(correct_foc_vgg)
            vgg_incorrect_foc.extend(incorrect_foc_vgg)
            
            svm_correct_res.extend(correct_res_svm)
            svm_incorrect_res.extend(incorrect_res_svm)
            svm_correct_foc.extend(correct_foc_svm)
            svm_incorrect_foc.extend(incorrect_foc_svm)

            xgb_correct_res.extend(correct_res_xgb)
            xgb_incorrect_res.extend(incorrect_res_xgb)
            xgb_correct_foc.extend(correct_foc_xgb)
            xgb_incorrect_foc.extend(incorrect_foc_xgb)
            
            
        ### print incorrectly classified rest images by svm, xgboost and vgg.
        imgs_path = results_face + "/classified_images/"
        if not os.path.exists(imgs_path):
            os.makedirs(imgs_path) 
            
        prediction_images(vgg_correct_res, vgg_incorrect_res, vgg_correct_foc, vgg_incorrect_foc, dataset_224, imgs_path, method1, n) 
        prediction_images(svm_correct_res, svm_incorrect_res, svm_correct_foc, svm_incorrect_foc, dataset_224, imgs_path, method2, n)
        prediction_images(xgb_correct_res, xgb_incorrect_res, xgb_correct_foc, xgb_incorrect_foc, dataset_224, imgs_path, method3, n)
    
    ###########################################################################
    #### results of facial images based methods
    ### results of accuracy
    acc_ensem_mean_f0 = np.array(acc_ensem_fac0).mean()
    acc_ensem_std_f0 = np.array(acc_ensem_fac0).std()
    acc_ensem_conf_f0 = st.t.interval(0.95, len(acc_ensem_fac0)-1, loc=np.mean(acc_ensem_fac0), scale=st.sem(acc_ensem_fac0))
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
    rest_pre_ensem_mean_f0 = np.array(rest_pre_ensem_fac0).mean()
    rest_pre_ensem_std_f0 = np.array(rest_pre_ensem_fac0).std() 
    rest_pre_ensem_conf_f0 = st.t.interval(0.95, len(rest_pre_ensem_fac0)-1, loc=np.mean(rest_pre_ensem_fac0), scale=st.sem(rest_pre_ensem_fac0)) 
    focus_pre_ensem_mean_f0 = np.array(focus_pre_ensem_fac0).mean()
    focus_pre_ensem_std_f0 = np.array(focus_pre_ensem_fac0).std()
    focus_pre_ensem_conf_f0 = st.t.interval(0.95, len(focus_pre_ensem_fac0)-1, loc=np.mean(focus_pre_ensem_fac0), scale=st.sem(focus_pre_ensem_fac0))
    
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
    rest_rec_ensem_mean_f0 = np.array(rest_rec_ensem_fac0).mean()
    rest_rec_ensem_std_f0 = np.array(rest_rec_ensem_fac0).std() 
    rest_rec_ensem_conf_f0 = st.t.interval(0.95, len(rest_rec_ensem_fac0)-1, loc=np.mean(rest_rec_ensem_fac0), scale=st.sem(rest_rec_ensem_fac0))
    focus_rec_ensem_mean_f0 = np.array(focus_rec_ensem_fac0).mean()
    focus_rec_ensem_std_f0 = np.array(focus_rec_ensem_fac0).std()
    focus_rec_ensem_conf_f0 = st.t.interval(0.95, len(focus_rec_ensem_fac0)-1, loc=np.mean(focus_rec_ensem_fac0), scale=st.sem(focus_rec_ensem_fac0))
    
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
    print("facial image equal weight method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, acc_ensem_mean_f0, acc_ensem_std_f0,acc_ensem_conf_f0[0],acc_ensem_conf_f0[1]))
    print("facial image optimal weight method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, acc_ensem_mean_f, acc_ensem_std_f,acc_ensem_conf_f[0],acc_ensem_conf_f[1]))
    print("facial image method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method1, acc_vgg_mean_f, acc_vgg_std_f, acc_vgg_conf_f[0],acc_vgg_conf_f[1]))
    print("facial image method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method2, acc_svm_mean_f, acc_svm_std_f, acc_svm_conf_f[0],acc_svm_conf_f[1]))
    print("facial image method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method3, acc_xgb_mean_f, acc_xgb_std_f, acc_xgb_conf_f[0],acc_xgb_conf_f[1]))
    
    ### print precision
    print("facial image equal weight method %s: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_pre_ensem_mean_f0, rest_pre_ensem_std_f0, rest_pre_ensem_conf_f0[0], rest_pre_ensem_conf_f0[1]))
    print("facial image equal weight method %s: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_pre_ensem_mean_f0, focus_pre_ensem_std_f0, focus_pre_ensem_conf_f0[0],focus_pre_ensem_conf_f0[1]))
    print("facial image optimal weight method %s: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_pre_ensem_mean_f, rest_pre_ensem_std_f, rest_pre_ensem_conf_f[0], rest_pre_ensem_conf_f[1]))
    print("facial image optimal weight method %s: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_pre_ensem_mean_f, focus_pre_ensem_std_f, focus_pre_ensem_conf_f[0],focus_pre_ensem_conf_f[1]))
    print("facial image method vgg: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_vgg_mean_f, rest_pre_vgg_std_f, rest_pre_vgg_conf_f[0],rest_pre_vgg_conf_f[1]))
    print("facial image method vgg: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_vgg_mean_f, focus_pre_vgg_std_f, focus_pre_vgg_conf_f[0], focus_pre_vgg_conf_f[1]))
    print("facial image method svm: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_svm_mean_f, rest_pre_svm_std_f, rest_pre_svm_conf_f[0], rest_pre_svm_conf_f[1]))
    print("facial image method svm: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_svm_mean_f, focus_pre_svm_std_f, focus_pre_svm_conf_f[0], focus_pre_svm_conf_f[1]))   
    print("facial image method xgb: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_xgb_mean_f, rest_pre_xgb_std_f, rest_pre_xgb_conf_f[0], rest_pre_xgb_conf_f[1]))
    print("facial image method xgb: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_xgb_mean_f, focus_pre_xgb_std_f, focus_pre_xgb_conf_f[0], focus_pre_xgb_conf_f[1]))

    ### print recall
    print("facial image equal weight method %s: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_rec_ensem_mean_f0, rest_rec_ensem_std_f0, rest_rec_ensem_conf_f0[0], rest_rec_ensem_conf_f0[1]))
    print("facial image equal weight method %s: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_rec_ensem_mean_f0, focus_rec_ensem_std_f0, focus_rec_ensem_conf_f0[0],focus_rec_ensem_conf_f0[1]))
    print("facial image optimal weight method %s: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_rec_ensem_mean_f, rest_rec_ensem_std_f, rest_rec_ensem_conf_f[0], rest_rec_ensem_conf_f[1]))
    print("facial image optimal weight method %s: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_rec_ensem_mean_f, focus_rec_ensem_std_f, focus_rec_ensem_conf_f[0],focus_rec_ensem_conf_f[1]))
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
        f.write("facial image equal weight method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, acc_ensem_mean_f0, acc_ensem_std_f0,acc_ensem_conf_f0[0],acc_ensem_conf_f0[1]))
        f.write("facial image optimal weight method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, acc_ensem_mean_f, acc_ensem_std_f,acc_ensem_conf_f[0],acc_ensem_conf_f[1])) 
        f.write("facial image method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method1, acc_vgg_mean_f, acc_vgg_std_f, acc_vgg_conf_f[0],acc_vgg_conf_f[1]))
        f.write("facial image method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method2, acc_svm_mean_f, acc_svm_std_f, acc_svm_conf_f[0],acc_svm_conf_f[1]))
        f.write("facial image method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method3, acc_xgb_mean_f, acc_xgb_std_f, acc_xgb_conf_f[0],acc_xgb_conf_f[1]))
        f.write('--'*40 +'\n')
        ### write precision
        f.write("facial image equal weight method %s: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_pre_ensem_mean_f0, rest_pre_ensem_std_f0, rest_pre_ensem_conf_f0[0], rest_pre_ensem_conf_f0[1]))
        f.write("facial image equal weight method %s: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_pre_ensem_mean_f0, focus_pre_ensem_std_f0, focus_pre_ensem_conf_f0[0],focus_pre_ensem_conf_f0[1]))
        f.write("facial image optimal weight method %s: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_pre_ensem_mean_f, rest_pre_ensem_std_f, rest_pre_ensem_conf_f[0], rest_pre_ensem_conf_f[1]))
        f.write("facial image optimal weight method %s: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_pre_ensem_mean_f, focus_pre_ensem_std_f, focus_pre_ensem_conf_f[0],focus_pre_ensem_conf_f[1]))
        f.write("facial image method vgg: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_vgg_mean_f, rest_pre_vgg_std_f, rest_pre_vgg_conf_f[0],rest_pre_vgg_conf_f[1]))
        f.write("facial image method vgg: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_vgg_mean_f, focus_pre_vgg_std_f, focus_pre_vgg_conf_f[0], focus_pre_vgg_conf_f[1]))
        f.write("facial image method svm: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_svm_mean_f, rest_pre_svm_std_f, rest_pre_svm_conf_f[0], rest_pre_svm_conf_f[1]))
        f.write("facial image method svm: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_svm_mean_f, focus_pre_svm_std_f, focus_pre_svm_conf_f[0], focus_pre_svm_conf_f[1]))   
        f.write("facial image method xgb: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_xgb_mean_f, rest_pre_xgb_std_f, rest_pre_xgb_conf_f[0], rest_pre_xgb_conf_f[1]))
        f.write("facial image method xgb: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_xgb_mean_f, focus_pre_xgb_std_f, focus_pre_xgb_conf_f[0], focus_pre_xgb_conf_f[1]))

        f.write('--'*40 +'\n')
        ### write recall
        f.write("facial image equal weight method %s: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_rec_ensem_mean_f0, rest_rec_ensem_std_f0, rest_rec_ensem_conf_f0[0], rest_rec_ensem_conf_f0[1]))
        f.write("facial image equal weight method %s: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_rec_ensem_mean_f0, focus_rec_ensem_std_f0, focus_rec_ensem_conf_f0[0],focus_rec_ensem_conf_f0[1]))
        f.write("facial image optimal weight method %s: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_rec_ensem_mean_f, rest_rec_ensem_std_f, rest_rec_ensem_conf_f[0], rest_rec_ensem_conf_f[1]))
        f.write("facial image optimal weight method %s: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_rec_ensem_mean_f, focus_rec_ensem_std_f, focus_rec_ensem_conf_f[0],focus_rec_ensem_conf_f[1]))
        f.write("facial image method vgg: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_vgg_mean_f, rest_rec_vgg_std_f, rest_rec_vgg_conf_f[0],rest_rec_vgg_conf_f[1]))
        f.write("facial image method vgg: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_vgg_mean_f, focus_rec_vgg_std_f, focus_rec_vgg_conf_f[0], focus_rec_vgg_conf_f[1]))
        f.write("facial image method svm: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_svm_mean_f, rest_rec_svm_std_f, rest_rec_svm_conf_f[0], rest_rec_svm_conf_f[1]))
        f.write("facial image method svm: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_svm_mean_f, focus_rec_svm_std_f, focus_rec_svm_conf_f[0], focus_rec_svm_conf_f[1]))   
        f.write("facial image method xgb: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_xgb_mean_f, rest_rec_xgb_std_f, rest_rec_xgb_conf_f[0], rest_rec_xgb_conf_f[1]))
        f.write("facial image method xgb: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_xgb_mean_f, focus_rec_xgb_std_f, focus_rec_xgb_conf_f[0], focus_rec_xgb_conf_f[1]))    
        f.write('--'*40 +'\n')
        f.write('facial image based: the size of method vgg is {} bytes, and the training time is {} \n'.format(vgg_size, train_time_vgg))
        f.write('facial image based: the size of method vgg is {} bytes, and the testing time is {} \n'.format(vgg_size, test_time_vgg))
        f.write('facial image based: the size of method svm is {} bytes, and the training time is {} \n'.format(svm_size, train_time_svm))
        f.write('facial image based: the size of method svm is {} bytes, and the testing time is {} \n'.format(svm_size, test_time_svm))
        f.write('facial image based: the size of method xgb is {} bytes, and the training time is {} \n'.format(xgb_size, train_time_xgb))
        f.write('facial image based: the size of method xgb is {} bytes, and the testing time is {} \n'.format(xgb_size, test_time_xgb))

    
    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='')
    parser.add_argument('--num_epochs', type=int, default=150,
                        help='')
    parser.add_argument('--method', type=str, default='ensemble',
                        help='') 
    parser.add_argument('--test', type=bool, default=False,
                        help='')      
    parser.add_argument('--results', type=str, default='./facial_image_results/',
                        help='')  
    parser.add_argument('--cuda', type=bool,  default=True,
                        help='use CUDA')  
    parser.add_argument('--device_id', type=str, default='0')
    
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id  
 
    # seed_everything(args.seed)
    
    torch.cuda.is_available()
    
    results_face = args.results+'facial_image/'

    if not os.path.exists(results_face):
        os.makedirs(results_face)
   

    main(args, results_face)

    
