#!/usr/bin/env python3
# -*- coding: utf-8 -*-
   
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

def polt_confusion(model, x_test, y_test, class_names,results_p):
    np.set_printoptions(precision=2)
    
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

    plt.savefig(results_p+"SVM_cofusion_matrix.png")
        
    plt.show()

def svm_f(batch_size, frame_size, results_p, method):
    image_dir = './data/images_{}x{}/'.format(frame_size[0], frame_size[1])
    
    rest_csv = 'rest.csv'
    focus_csv = 'focus.csv'
    results_f = results_path + '{}_restults.txt'.format(method)
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

  
    param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
    
    svc=svm.SVC(probability=True)
    model=GridSearchCV(svc,param_grid)
    
    train_loader, test_loader, img_dataset = create_datasets(batch_size,transform, image_dir, rest_csv, focus_csv)
      
    x = []
    y = []
    
    for i in range(len(img_dataset)):
        x.append(img_dataset[i][0].numpy().flatten())
        y.append(img_dataset[i][1].numpy())
    
    x = np.asarray(x)
    y = np.asarray(y)
    # print(x.shape)
    # print(y.shape)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)
    
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    
    print("The  results of SVM for stress detection ... ...\n")
    print("The predicted Data is :")
    print(y_pred)
    print("The actual data is:")
    print(np.array(y_test))
    print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")
    
    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    with open(results_f, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        f.writelines("The test accracy of {} is {} \n".format(method, accuracy_score(y_pred,y_test)))
    
    class_names = ['rest', 'focus']
    polt_confusion(model, x_test, y_test, class_names,results_p)
    
   
if __name__=="__main__":
    batch_size = 32
    frame_size = (28,28)
    
    svm_f(batch_size, frame_size)


    
    