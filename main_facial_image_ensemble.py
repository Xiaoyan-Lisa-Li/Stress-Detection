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
    y_prob = []

    for x_test, test_y in test_loader:
        total_num += len(test_y)
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
    

    if method == '2d_cnn':
        fig_path = results_path + "/2d_cnn_figures/"
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
    elif method == 'pre_vgg':
        fig_path = results_path + "/pretain_vgg_figures/"
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
                
    plot_confusion2(y_true, y_pred, method, fig_path, fold, n, labels = [0,1])
    
    return  y_true, y_prob, acc, rest_precision, focus_precision

def predict_svm(model, x_test, y_test, results_p, method, fold, n):
    rest_true = 0
    rest_false = 0
    focus_true = 0
    focus_false = 0  
    focus_pre_ls = []
    rest_pre_ls = []
    acc_ls = []
    results_f = results_p + '{}_restults.txt'.format(method)
    
    y_pred = model.predict_proba(x_test)
    y_prob = y_pred[:,1]
    y_pred = np.argmax(y_pred, axis=1).tolist()
    print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")
  
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
    
    rest_pre_ls.append(rest_precision)
    focus_pre_ls.append(focus_precision)
    acc_ls.append(acc)
    
    print("accuracy is %0.4f \n" % (acc))
    print("focus has %0.4f precision\n" % (focus_precision))
    print("rest has %0.4f precision \n" % (rest_precision))   

    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    with open(results_f, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        f.writelines("accuracy is %0.4f \n" % (acc))
        f.writelines("focus has %0.4f precision\n" % (focus_precision))
        f.writelines("rest has %0.4f precision \n" % (rest_precision))
        
    class_names = ['rest', 'focus']  
    polt_confusion(model, x_test, y_test, class_names, results_p, fold, n)
    
    return y_prob, acc, rest_precision, focus_precision

def train_model(model, train_loader, test_loader, num_epochs, checkpoint, results_path, method, fold, n):
    
    checkpoint_path = results_path + checkpoint
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        
  
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
        
        
        # if (epoch+1) %10 == 0:
        #     save_path = checkpoint_path + 'model{}_repeat{}_fold{}.pth'.format(epoch,n,fold)  
        #     save_objs(model, epoch, avg_loss, optimizer, save_path)

    
    plot_loss(method, train_losses, val_losses, results_path)
    
    

def main(args):
       
    image_path1 = './data/images_28x28/'
    image_path2 = './data/images_224x224/'
    
    image_dir1 = image_path1 + 'images/'
    image_dir2 = image_path2 + 'images/'
    img_csv = 'image.csv'
    
    check_path_cnn = args.results + 'checkpt_cnn'
    check_path_vgg = args.results + 'checkpt_vgg'
    
    k_folds = 5
    repeat = 3
    
      # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

   
    ## both trian and test data are from all 25 videos
    _, _, dataset_28 = create_datasets(args.batch_size,transform, image_path1, image_dir1, img_csv)
    _, _, dataset_224 = create_datasets(args.batch_size,transform, image_path2, image_dir2, img_csv)
     
    acc_en_ls = []
    acc_cnn_ls = []
    acc_vgg_ls = []
    acc_svm_ls = []
    rest_prec_ls = []
    focus_prec_ls = []
    rest_cnn_ls = []
    focus_cnn_ls = []
    rest_vgg_ls = []
    focus_vgg_ls = []
    rest_svm_ls = []
    focus_svm_ls = []

    rest_true = 0
    rest_false = 0
    focus_true = 0
    focus_false = 0
    
    
    x_28 = []
    y_28 = []  
    for i in range(len(dataset_28)): 
        x_28.append(dataset_28[i][0].numpy().flatten())
        y_28.append(dataset_28[i][1].numpy())
    x_28 = np.asarray(x_28)
    y_28 = np.asarray(y_28)
   
    # x_224 = []
    # y_224 = []    
    # for i in range(len(dataset_28)):    
    #     x_224.append(dataset_224[i][0].numpy().flatten())
    #     y_224.append(dataset_224[i][1].numpy())
    # x_224 = np.asarray(x_224)
    # y_224 = np.asarray(y_224)
    
    for n in range(repeat):
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset_28)):
            print(f'FOLD {fold}')
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
           
            ### create trainset and testset for 2D CNN
            train_loader_28 = torch.utils.data.DataLoader(
                              dataset_28, batch_size=args.batch_size, sampler=train_subsampler)
            testset = torch.utils.data.Subset(dataset_28, test_ids.tolist())
            test_loader_28 = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    
            ### create trainset and testset for pretained vgg
            train_loader_224 = torch.utils.data.DataLoader(
                              dataset_224, batch_size=args.batch_size, sampler=train_subsampler)
            testset = torch.utils.data.Subset(dataset_224, test_ids.tolist())
            test_loader_224 = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False) 
            
            ### create train set and test set for svm
            x_train_28, y_train_28 = x_28[train_ids], y_28[train_ids]
            x_test_28, y_test_28 = x_28[test_ids], y_28[test_ids]           
            x_train_28, y_train_28 = shuffle(x_train_28, y_train_28) ## only shuffle train dataset

            
            method1 = '2d_cnn'
            model_cnn=CNN_2d().cuda()
            model_cnn.apply(reset_weights)
            num_epochs = 120
            train_model(model_cnn, train_loader_28, test_loader_28, num_epochs, check_path_cnn, args.results, method1, fold, n)
            y_cnn, pred_cnn, acc_cnn, rest_pre_cnn, focus_pre_cnn  = predict(model_cnn, test_loader_28, check_path_cnn, num_epochs-1, method1, args.results, fold, n)
            
            method2 = 'pre_vgg'
            model_vgg = alexnet().cuda()
            reset_weights_vgg(model_vgg)
            num_epochs = 200
            train_model(model_vgg, train_loader_224, test_loader_224, num_epochs, check_path_vgg, args.results, method2, fold, n)
            y_vgg, pred_vgg, acc_vgg, rest_pre_vgg, focus_pre_vgg = predict(model_vgg, test_loader_224, check_path_vgg, num_epochs-1, method2, args.results, fold, n)
            
            method3 = 'svm'
            model_svm= svm.SVC(kernel='poly', probability=True)
            # model= svm.SVC(kernel='rbf', C=5.0, probability=True)
            model_svm.fit(x_train_28,y_train_28)
            pred_svm, acc_svm, rest_pre_svm, focus_pre_svm = predict_svm(model_svm, x_test_28, y_test_28, args.results, method3, fold, n)
            
            print('y_cnn == y_vgg ?', y_cnn==y_vgg)

            y_en = []
            for i in range(len(y_test_28)):
                pred_ave = (pred_cnn[i] + pred_vgg[i] + pred_svm[i])/3
                if pred_ave >= 0.5:
                    y_en.append(1)
                else:
                    y_en.append(0)
                
            
            ## calculate how many samples are predicted correctly.
            for t, e in zip(y_test_28, y_en):
                if t == e and t.item() == 0:
                    rest_true += 1
                elif t != e and t.item() == 0:
                    rest_false += 1
                elif t == e and t.item() == 1:
                    focus_true += 1
                else:
                    focus_false += 1
        
            rest_precision = rest_true/(rest_true+focus_false)
            focus_precision = focus_true/(focus_true+rest_false)
            acc_en = accuracy_score(y_test_28, y_en)
            
            print('ensemble accuracy1 is {}'.format(acc_en))
    
            rest_prec_ls.append(rest_precision)
            focus_prec_ls.append(focus_precision)
            rest_cnn_ls.append(rest_pre_cnn)
            focus_cnn_ls.append(focus_pre_cnn)
            rest_vgg_ls.append(rest_pre_vgg)
            focus_vgg_ls.append(focus_pre_vgg)
            rest_svm_ls.append(rest_pre_svm)
            focus_svm_ls.append(focus_pre_svm)            
            acc_en_ls.append(acc_en)
            acc_cnn_ls.append(acc_cnn)
            acc_vgg_ls.append(acc_vgg)
            acc_svm_ls.append(acc_svm)
            
                                
            fig_path = results_path + "/emsemble_figures/"
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            plot_confusion2(y_vgg, y_en, args.method, fig_path, fold, n, labels = [0,1])              
            
    acc_en_mean = np.array(acc_en_ls).mean()
    acc_en_std = np.array(acc_en_ls).std()
    acc_cnn_mean = np.array(acc_cnn_ls).mean()
    acc_cnn_std = np.array(acc_cnn_ls).std()
    acc_vgg_mean = np.array(acc_vgg_ls).mean()
    acc_vgg_std = np.array(acc_vgg_ls).std()
    acc_svm_mean = np.array(acc_svm_ls).mean()
    acc_svm_std = np.array(acc_svm_ls).std()

    rest_mean = np.array(rest_prec_ls).mean()
    rest_std = np.array(rest_prec_ls).std()  
    focus_mean = np.array(focus_prec_ls).mean()
    focus_std = np.array(focus_prec_ls).std()

    rest_cnn_mean = np.array(rest_cnn_ls).mean()
    rest_cnn_std = np.array(rest_cnn_ls).std()  
    focus_cnn_mean = np.array(focus_cnn_ls).mean()
    focus_cnn_std = np.array(focus_cnn_ls).std()

    rest_vgg_mean = np.array(rest_vgg_ls).mean()
    rest_vgg_std = np.array(rest_vgg_ls).std()  
    focus_vgg_mean = np.array(focus_vgg_ls).mean()
    focus_vgg_std = np.array(focus_vgg_ls).std()

    rest_svm_mean = np.array(rest_svm_ls).mean()
    rest_svm_std = np.array(rest_svm_ls).std()  
    focus_svm_mean = np.array(focus_svm_ls).mean()
    focus_svm_std = np.array(focus_svm_ls).std()    
    
    
    print("Method %s: %0.4f accuracy with a standard deviation of %0.4f" % (args.method, acc_en_mean, acc_en_std))
    print("Method %s: %0.4f accuracy with a standard deviation of %0.4f" % (method1, acc_cnn_mean, acc_cnn_std))
    print("Method %s: %0.4f accuracy with a standard deviation of %0.4f" % (method2, acc_vgg_mean, acc_vgg_std))
    print("Method %s: %0.4f accuracy with a standard deviation of %0.4f" % (method3, acc_svm_mean, acc_svm_std))
    print("Method %s: %0.4f rest precision with a standard deviation of %0.4f" % (args.method, rest_mean, rest_std))
    print("Method %s: %0.4f focus precision with a standard deviation of %0.4f" % (args.method, focus_mean, focus_std))
    print("Method 2dcnn: %0.4f rest precision with a standard deviation of %0.4f \n" % (rest_cnn_mean, rest_cnn_std))
    print("Method 2dcnn: %0.4f focus precision with a standard deviation of %0.4f \n" % (focus_cnn_mean, focus_cnn_std))
    print("Method vgg: %0.4f rest precision with a standard deviation of %0.4f \n" % (rest_vgg_mean, rest_vgg_std))
    print("Method vgg: %0.4f focus precision with a standard deviation of %0.4f \n" % (focus_vgg_mean, focus_vgg_std))
    print("Method svm: %0.4f rest precision with a standard deviation of %0.4f \n" % (rest_svm_mean, rest_svm_std))
    print("Method svm: %0.4f focus precision with a standard deviation of %0.4f \n" % (focus_svm_mean, focus_svm_std))   
    
    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    results_f = args.results + '{}_restults.txt'.format(args.method)
    with open(results_f, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        f.write("Method %s: %0.4f accuracy with a standard deviation of %0.4f \n" % (args.method, acc_en_mean, acc_en_std))
        f.write("Method %s: %0.4f accuracy with a standard deviation of %0.4f \n" % (method1, acc_cnn_mean, acc_cnn_std))
        f.write("Method %s: %0.4f accuracy with a standard deviation of %0.4f \n" % (method2, acc_vgg_mean, acc_vgg_std))
        f.write("Method %s: %0.4f accuracy with a standard deviation of %0.4f \n" % (method3, acc_svm_mean, acc_svm_std))    
        f.write("Method %s: %0.4f rest precision with a standard deviation of %0.4f \n" % (args.method, rest_mean, rest_std))
        f.write("Method %s: %0.4f focus precision with a standard deviation of %0.4f \n" % (args.method, focus_mean, focus_std))
        f.write("Method 2dcnn: %0.4f rest precision with a standard deviation of %0.4f \n" % (rest_cnn_mean, rest_cnn_std))
        f.write("Method 2dcnn: %0.4f focus precision with a standard deviation of %0.4f \n" % (focus_cnn_mean, focus_cnn_std))
        f.write("Method vgg: %0.4f rest precision with a standard deviation of %0.4f \n" % (rest_vgg_mean, rest_vgg_std))
        f.write("Method vgg: %0.4f focus precision with a standard deviation of %0.4f \n" % (focus_vgg_mean, focus_vgg_std))
        f.write("Method svm: %0.4f rest precision with a standard deviation of %0.4f \n" % (rest_svm_mean, rest_svm_std))
        f.write("Method svm: %0.4f focus precision with a standard deviation of %0.4f \n" % (focus_svm_mean, focus_svm_std))        
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='')
    parser.add_argument('--num_epochs', type=int, default=300,
                        help='')
    parser.add_argument('--method', type=str, default='ensemble',
                        help='')    
    parser.add_argument('--seed', type=int, default=2021,
                        help='')    
    parser.add_argument('--results', type=str, default='./facial_image_results/ensemble_results/',
                        help='')  
    
    args = parser.parse_args()
    
    # seed_everything(args.seed)
    
    torch.cuda.is_available()
    
    if not os.path.exists(args.results):
        os.makedirs(args.results)
    

    main(args)

    
