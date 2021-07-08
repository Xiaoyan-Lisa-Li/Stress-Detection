#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from data_processing import *
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from datetime import datetime
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from numpy import std, mean

def polt_confusion(model, x_test, y_test, class_names,results_p, i):
    np.set_printoptions(precision=4)
    
    fig_path = result_p + '/svm_confusion/'
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

    plt.savefig(fig_path+"SVM_cofusion_matrix_{}.png".format(i))
        
    plt.show()

def svm_f(batch_size, frame_size, results_p, method):
    image_dir = './data/images_{}x{}/'.format(frame_size[0], frame_size[1])
    
    rest_csv = 'rest.csv'
    focus_csv = 'focus.csv'
    results_f = results_p + '{}_restults.txt'.format(method)
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)
    param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['linear','poly']}
    train_loader, test_loader, img_dataset = create_datasets(batch_size,transform, image_dir, rest_csv, focus_csv)
    
    svc=svm.SVC(kernel='linear')
    # model=GridSearchCV(svc,param_grid)
    model = svc
      
    x = []
    y = []
    
    for i in range(len(img_dataset)):
        x.append(img_dataset[i][0].numpy().flatten())
        y.append(img_dataset[i][1].numpy())
    
    x = np.asarray(x)
    y = np.asarray(y)
    # print(x.shape)
    # print(y.shape)
    scores_acc = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # focus_precision = cross_val_score(model, x, 1-y, scoring='precision', cv=cv, n_jobs=-1)
    # rest_precision = cross_val_score(model, x, y, scoring='precision', cv=cv, n_jobs=-1)
    
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores_acc.mean(), scores_acc.std()))
    # print("focus has %0.2f precision with a standard deviation of %0.2f" % (focus_precision.mean(), focus_precision.std()))
    # print("rest has %0.2f precision with a standard deviation of %0.2f" % (rest_precision.mean(), rest_precision.std()))
    
    
    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    with open(results_f, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        # f.writelines("The test accracy of {} is {} \n".format(method, accuracy_score(y_pred,y_test)))
        f.writelines("%0.2f accuracy with a standard deviation of %0.2f \n" % (scores_acc.mean(), scores_acc.std()))
        # f.writelines("focus has %0.2f precision with a standard deviation of %0.2f \n" % (focus_precision.mean(), focus_precision.std()))
        # f.writelines("rest has %0.2f precision with a standard deviation of %0.2f \n" % (rest_precision.mean(), rest_precision.std()))

    for i in range(10):
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,stratify=y)   
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
        print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")
        class_names = ['rest', 'focus']
        polt_confusion(model, x_test, y_test, class_names,results_p,i)
    
   
if __name__=="__main__":
    batch_size = 32
    frame_size = (28,28)
    result_p = './method1_results/'
    svm_f(batch_size, frame_size, result_p, method = 'svm')


    
    