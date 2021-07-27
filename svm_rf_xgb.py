#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from facial_data_process import create_datasets
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from datetime import datetime
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
from numpy import std, mean
from ecg_data_process import create_ecg_data
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from utils import plot_confusion


def svm_rf_xgb_f(batch_size, frame_size, method):
  
    # param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
    # svc=svm.SVC()
    # model=GridSearchCV(svc,param_grid)
    
    if method== 'svm':
        results_p = './facial_image_results/svm/'
        model= svm.SVC(kernel='poly', C=1., probability=True)
        # model= svm.SVC(kernel='rbf', C=5., probability=True)     
    
    if method == 'rf':
        results_p = './facial_image_results/random_forest/'
        model = RandomForestClassifier(n_estimators=200, random_state=0)
    if method == 'xgb':
        results_p = './facial_image_results/xgboost/'
        # model = xgb.XGBClassifier(objective="binary:logistic", learning_rate=0.5,max_depth=5,n_estimators=200, random_state=42)
        model = xgb.XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0,  subsample=0.8,\
                                  colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
            
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    kf = KFold(n_splits=5, shuffle=True)
    
    if not os.path.exists(results_p):
        os.makedirs(results_p)
        
    image_path = './data/images_{}x{}_2/'.format(frame_size[0],frame_size[1])
    image_dir = image_path + 'images/'
    img_csv = 'image.csv'
    
    results_f = results_p + '{}_restults.txt'.format(method)
    class_names = ['rest', 'focus']

    train_loader, test_loader, img_dataset = create_datasets(batch_size,transform, image_path, image_dir, img_csv)
      
    x = []
    y = []
    
    for i in range(len(img_dataset)):    
        x.append(img_dataset[i][0].numpy().flatten())
        y.append(img_dataset[i][1].numpy())
        
    # print(img_dataset[i][0].numpy().flatten())
    x = np.asarray(x)
    y = np.asarray(y)
    
    rest_true = 0
    rest_false = 0
    focus_true = 0
    focus_false = 0  
    
    focus_pre_ls = []
    rest_pre_ls = []
    acc_ls = []
    for i in range(3):
        k = 0
        
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            x_train, y_train = shuffle(x_train, y_train)
            ### x_test, y_test = shuffle(x_test, y_test)
            
            model.fit(x_train,y_train)
            y_pred = model.predict_proba(x_test)
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
                f.writelines("repeat {}, fold {} \n".format(i, k))
                f.writelines("accuracy is %0.4f \n" % (acc))
                f.writelines("focus has %0.4f precision\n" % (focus_precision))
                f.writelines("rest has %0.4f precision \n" % (rest_precision))
             
            k += 1    
            plot_confusion(model, x_test, y_test, class_names,results_p, method, i, k)
    
    acc_array = np.array(acc_ls)    
    rest_pre_array = np.array(rest_pre_ls)    
    focus_pre_array = np.array(focus_pre_ls)  
    print("%0.4f accuracy with a standard deviation of %0.4f" % (acc_array.mean(), acc_array.std()))
    print("focus has %0.4f precision with a standard deviation of %0.4f \n" % (focus_pre_array.mean(), focus_pre_array.std()))
    print("rest has %0.4f precision with a standard deviation of %0.4f \n" % (rest_pre_array.mean(), rest_pre_array.std()))   

    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    with open(results_f, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        f.writelines("%0.4f accuracy with a standard deviation of %0.4f \n" % (acc_array.mean(), acc_array.std()))
        f.writelines("focus has %0.4f precision with a standard deviation of %0.4f \n" % (focus_pre_array.mean(), focus_pre_array.std()))
        f.writelines("rest has %0.4f precision with a standard deviation of %0.4f \n" % (rest_pre_array.mean(), rest_pre_array.std()))
                
    
def svm_ecg(method, seconds):
    
    if method== 'svm':
        results_ecg = './ecg_results/svm/'
        model= svm.SVC(kernel='poly', degree = 2, C =500)
        # model= svm.SVC(kernel='rbf', C=5.)     
    
    if method == 'rf':
        results_ecg = './ecg_results/random_forest/'
        model = RandomForestClassifier(n_estimators=300, random_state=0)
    if method == 'xgb':
        results_ecg = './ecg_results/xgboost/'
        # model = xgb.XGBClassifier(objective="binary:logistic", learning_rate=0.1, max_depth=30, n_estimators=300, random_state=42)
        model = xgb.XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0,  subsample=0.8,\
                                  colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
    
    if not os.path.exists(results_ecg):
        os.makedirs(results_ecg)
        
    results_f = results_ecg + '{}_restults.txt'.format(method)
    class_names = ['rest', 'focus']
    
    x,y = create_ecg_data(time_s = seconds, window_s=3)

    
    
    kf = KFold(n_splits=5, shuffle=True)
    
    
    rest_true = 0
    rest_false = 0
    focus_true = 0
    focus_false = 0  
    
    focus_pre_ls = []
    rest_pre_ls = []
    acc_ls = []
    for i in range(3):
        k = 0      
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            x_train, y_train = shuffle(x_train, y_train)
            x_test, y_test = shuffle(x_test, y_test)
            
            model.fit(x_train,y_train)
            y_pred=model.predict(x_test)
            
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
                f.writelines("repeat {}, fold {} \n".format(i, k))
                f.writelines("accuracy is %0.4f \n" % (acc))
                f.writelines("focus has %0.4f precision\n" % (focus_precision))
                f.writelines("rest has %0.4f precision \n" % (rest_precision))
             
            k += 1    
            plot_confusion(model, x_test, y_test, class_names,results_ecg, method, i, k)
        
    
    acc_array = np.array(acc_ls)    
    rest_pre_array = np.array(rest_pre_ls)    
    focus_pre_array = np.array(focus_pre_ls)  
    print("%0.4f accuracy with a standard deviation of %0.4f" % (acc_array.mean(), acc_array.std()))
    print("focus has %0.4f precision with a standard deviation of %0.4f \n" % (focus_pre_array.mean(), focus_pre_array.std()))
    print("rest has %0.4f precision with a standard deviation of %0.4f \n" % (rest_pre_array.mean(), rest_pre_array.std()))   

    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    with open(results_f, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        f.writelines("%0.4f accuracy with a standard deviation of %0.4f \n" % (acc_array.mean(), acc_array.std()))
        f.writelines("focus has %0.4f precision with a standard deviation of %0.4f \n" % (focus_pre_array.mean(), focus_pre_array.std()))
        f.writelines("rest has %0.4f precision with a standard deviation of %0.4f \n" % (rest_pre_array.mean(), rest_pre_array.std()))
   
if __name__=="__main__":
    batch_size = 64
    frame_size = (28,28)
    method = ['svm','rf','xgb']
    svm_rf_xgb_f(batch_size, frame_size, method = method[2])
    # svm_ecg(method = 'xgb', seconds=360)
    

    
    