#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import csv
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
import math


class FacialImagesDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        print(self.root_dir)
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        
        if self.transform:
            image = self.transform(image)
            
        return (image, y_label)
    

def creat_csv(csv_file, path, label):
    '''
    since use model = 'a'  in "with open(csv_file, 'a', newline='')" , when do testing, 
    each time we need to create new files: focus.csv and rest.csv. 
    Another ways to save data could be explored.
    '''
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image', 'label'])
        for root, dir_name, file_name in os.walk(path): 
            # print(root)
            # print(dir_name)
            # print(file_name)
            for name in file_name:
                # fullName = os.path.join(root, name)
                if label == 0:
                    writer.writerow([name, 0])
                else:
                    writer.writerow([name, 1])
                    

                        
def create_datasets(batch_size, transform, transform_t, image_dir, rest_csv, focus_csv):
    
    
    rest_data = FacialImagesDataset(csv_file = image_dir+rest_csv, root_dir = image_dir + 'rest/',
                                        transform = transform)
    
    # trans_rest = FacialImagesDataset(csv_file = image_dir+rest_csv, root_dir = image_dir + 'rest/',
    #                                     transform = transform_t)    
    # print(len(rest_dataset))
    
    focus_data = FacialImagesDataset(csv_file = image_dir+focus_csv, root_dir = image_dir + 'focus/',
                                        transform = transform)

    # trans_focus = FacialImagesDataset(csv_file = image_dir+focus_csv, root_dir = image_dir + 'focus/',
    #                                     transform = transform_t)
    
    
    img_dataset = torch.utils.data.ConcatDataset([rest_data, focus_data])
    print(img_dataset[0][0].size())
    
    # for i in range(len(img_dataset)):
    #     print(img_dataset[i][1])
   
    train_set, test_set = torch.utils.data.random_split(img_dataset, [round(len(img_dataset)*0.8),round(len(img_dataset)*0.2)]) 
    
    train_loader = DataLoader(dataset=train_set, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size = batch_size, shuffle=True)

 
    return train_loader, test_loader, img_dataset

def create_datasets2(batch_size, transform, transform_t, image_train_dir, image_test_dir, rest_csv, focus_csv):
    
    ###########################################################################
    ### create trainloader
    ###########################################################################
    train_rest = FacialImagesDataset(csv_file = image_train_dir + rest_csv, root_dir = image_train_dir + 'rest/',
                                        transform = transform)
  
    train_focus = FacialImagesDataset(csv_file = image_train_dir + focus_csv, root_dir = image_train_dir + 'focus/',
                                        transform = transform)

    train_set = torch.utils.data.ConcatDataset([train_rest, train_focus])

    train_loader = DataLoader(dataset=train_set, batch_size = batch_size, shuffle=True)
    
    ###########################################################################
    ### create testloader
    ###########################################################################
    test_rest = FacialImagesDataset(csv_file = image_test_dir + rest_csv, root_dir = image_test_dir + 'rest/',
                                        transform = transform)
    # print(len(rest_dataset))    
    test_focus = FacialImagesDataset(csv_file = image_test_dir + focus_csv, root_dir = image_test_dir + 'focus/',
                                        transform = transform)    
    
    test_set = torch.utils.data.ConcatDataset([test_rest, test_focus])
    test_loader = DataLoader(dataset=test_set, batch_size = batch_size, shuffle=True)

 
    return train_loader, test_loader



if __name__=="__main__":
    batch_size = 32
    frame_size = (28,28)
    label_rest = 0
    label_focus = 1
    
    image_dir = './data/images_{}x{}/'.format(frame_size[0],frame_size[1])
    image_train_dir = './data/images_train_{}x{}/'.format(frame_size[0],frame_size[1])
    image_test_dir = './data/images_test_{}x{}/'.format(frame_size[0],frame_size[1])
    
    rest_csv = 'rest.csv'
    focus_csv = 'focus.csv'

    
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    ### data augmentation
    transform_t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    
    # creat_csv(image_dir + rest_csv, image_dir +'rest/', label_rest)
    # creat_csv(image_dir + focus_csv, image_dir +'focus/',label_focus)

    # creat_csv(image_train_dir + rest_csv, image_train_dir +'rest/', label_rest)
    # creat_csv(image_train_dir + focus_csv, image_train_dir +'focus/',label_focus)    
    
    # creat_csv(image_test_dir + rest_csv, image_test_dir +'rest/', label_rest)
    # creat_csv(image_test_dir + focus_csv, image_test_dir +'focus/',label_focus)       
    
    ##########################################################################
    ### both trian and test data are from all 25 videos or 
    ### trian data is from first 21 videos and test data is from last 4 videos.
    ###########################################################################
    train_loader, test_loader, img_dataset = create_datasets(batch_size,transform, transform_t, image_dir, rest_csv, focus_csv)
    # train_loader, test_loader = create_datasets2(batch_size,transform, transform_t, image_train_dir, image_test_dir, rest_csv, focus_csv)
    
    
    # data_iter = iter(train_loader)   
    # images, labels = next(data_iter)
    # print('image size is',images.size()[0])
    # print('the corresponding label are: ', labels)
    
    # plt.imshow(torchvision.utils.make_grid(images, nrow=5).permute(1, 2, 0))
        
    # plt.show()
    
    
    from sklearn.model_selection import train_test_split
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score
    import numpy as np
    from sklearn.metrics import plot_confusion_matrix
    
    param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
    svc=svm.SVC(probability=True)
    model=GridSearchCV(svc,param_grid)
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
    print("The predicted Data is :")
    print(y_pred)
    print("The actual data is:")
    print(np.array(y_test))
    print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")
    
    np.set_printoptions(precision=2)
    class_names = ['rest', 'focus']

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
    
    plt.show()
       
    
    
   
  
    



