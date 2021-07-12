#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from facial_data_process import *
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from datetime import datetime
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from numpy import std, mean
from ecg_data_process import create_ecg_data

def polt_confusion(model, x_test, y_test, class_names,results, i):
    np.set_printoptions(precision=4)
    
    fig_path = results + 'svm_confusion/'
    
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    
    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(model, x_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)
    
        print(title)
        print(disp.confusion_matrix)
    
    print(fig_path)
    plt.savefig(fig_path+"SVM_cofusion_matrix_{}.png".format(i))
        
    plt.show()
    
def compute_confusion(kernel, C, x, y, results_p):
    model = svm.SVC(kernel=kernel, C=C)
    for i in range(15):
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,stratify=y)   
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
        print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")
        class_names = ['rest', 'focus']
        polt_confusion(model, x_test, y_test, class_names,results_p,i)

def svm_f(batch_size, frame_size, results_p, method):
    image_dir = './data/images_{}x{}/'.format(frame_size[0], frame_size[1])   
    rest_csv = 'rest.csv'
    focus_csv = 'focus.csv'
    results_f = results_p + '{}_restults.txt'.format(method)
    kernel = 'rbf' ### 'linear','poly'
    C = 1.0    
    
    # param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
    # svc=svm.SVC()
    # model=GridSearchCV(svc,param_grid)
    
    if kernel == 'rbf':
        C = 5.0
    model= svm.SVC(kernel=kernel, C=C)
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)

    train_loader, test_loader, img_dataset = create_datasets(batch_size,transform, image_dir, rest_csv, focus_csv)
      
    x = []
    y = []
    
    for i in range(len(img_dataset)):
        x.append(img_dataset[i][0].numpy().flatten())
        y.append(img_dataset[i][1].numpy())
    
    x = np.asarray(x)
    y = np.asarray(y)

    scores_acc = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
    rest_precision = cross_val_score(model, x, 1-y, scoring='precision', cv=cv, n_jobs=-1)
    focus_precision = cross_val_score(model, x, y, scoring='precision', cv=cv, n_jobs=-1)
    
    print("%0.4f accuracy with a standard deviation of %0.4f" % (scores_acc.mean(), scores_acc.std()))
    print("focus has %0.4f precision with a standard deviation of %0.4f \n" % (focus_precision.mean(), focus_precision.std()))
    print("rest has %0.4f precision with a standard deviation of %0.4f \n" % (rest_precision.mean(), rest_precision.std()))   
    
    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    with open(results_f, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        f.writelines("%0.4f accuracy with a standard deviation of %0.4f \n" % (scores_acc.mean(), scores_acc.std()))
        f.writelines("focus has %0.4f precision with a standard deviation of %0.4f \n" % (focus_precision.mean(), focus_precision.std()))
        f.writelines("rest has %0.4f precision with a standard deviation of %0.4f \n" % (rest_precision.mean(), rest_precision.std()))
        
    compute_confusion(kernel, C, x, y, results_p)
    
def svm_ecg(results_ecg, method):
    
    if not os.path.exists(results_ecg):
        os.makedirs(results_ecg)
        
    results_f = results_ecg + '{}_restults.txt'.format(method)
    x,y = create_ecg_data(time_s = 480)
    
    print(x)
    print(y)    
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
  
    model = svm.SVC(kernel='poly', degree = 2, C =500, max_iter=100000)
   
    
    
    scores_acc = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)

    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores_acc.mean(), scores_acc.std()))

    
    
    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    with open(results_f, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        f.writelines("%0.4f accuracy with a standard deviation of %0.4f \n" % (scores_acc.mean(), scores_acc.std()))

    for i in range(15):
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,stratify=y)   
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
        print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")
        class_names = ['rest', 'focus']
        polt_confusion(model, x_test, y_test, class_names,results_ecg,i)
    
   
if __name__=="__main__":
    batch_size = 64
    frame_size = (28,28)
    result_p = './facial_image_results/'
    result_ecg = './ecg_results/'
    svm_f(batch_size, frame_size, result_p, method = 'svm')
    # svm_ecg(result_ecg, method = 'svm')
    

    
    