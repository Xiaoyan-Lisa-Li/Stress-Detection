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

def polt_confusion(model, x_test, y_test, class_names,results, i, k):
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
   
        print('normalize = ',normalize)
        plt.savefig(fig_path+"SVM_cofusion_matrix_{}_{}_{}.png".format(normalize, i, k))
            
        plt.show()
    


def svm_f(batch_size, frame_size, results_p, method):
    image_path = './data/images_{}x{}_2/'.format(frame_size[0],frame_size[1])
    image_dir = image_path + 'images/'
    img_csv = 'image.csv'
    
    results_f = results_p + '{}_restults.txt'.format(method)
    class_names = ['rest', 'focus']
    kernel = 'poly' ### 'linear','poly'
    C = 1.0    
    
    # param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
    # svc=svm.SVC()
    # model=GridSearchCV(svc,param_grid)
    
    if kernel == 'rbf':
        C = 5.0
    model= svm.SVC(kernel=kernel, C=C, probability=True)
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    kf = KFold(n_splits=5, shuffle=True)

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
                f.writelines("accuracy is %0.4f \n" % (acc))
                f.writelines("focus has %0.4f precision\n" % (focus_precision))
                f.writelines("rest has %0.4f precision \n" % (rest_precision))
             
            k += 1    
            polt_confusion(model, x_test, y_test, class_names,results_p, i, k)
    
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
                
    
def svm_ecg(results_ecg, method):
    
    if not os.path.exists(results_ecg):
        os.makedirs(results_ecg)
        
    results_f = results_ecg + '{}_restults.txt'.format(method)
    class_names = ['rest', 'focus']
    x,y = create_ecg_data(time_s = 540,window_s=3)
    
    print(x)
    print(y) 
       
    # cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    model= svm.SVC(kernel='poly', degree = 2, C =500)
    # model = svm.SVC(kernel='rbf', C=5)
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
                f.writelines("accuracy is %0.4f \n" % (acc))
                f.writelines("focus has %0.4f precision\n" % (focus_precision))
                f.writelines("rest has %0.4f precision \n" % (rest_precision))
             
            k += 1    
            polt_confusion(model, x_test, y_test, class_names,results_ecg, i, k)
    
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
    result_img = './facial_image_results/'
    result_ecg = './ecg_results/'
    svm_f(batch_size, frame_size, result_img, method = 'svm')
    # svm_ecg(result_ecg, method = 'svm')
    

    
    