#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 22:27:21 2021

@author: Xiaoyan
"""

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
    since use model = 'a'  in "with open(csv_file, 'a', newline='')" , when do testing, we need to delect the generated file: focus.csv and rest.csv. 
    
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
                        
def create_datasets():
    image_dir = './data/images/'
    rest_csv = image_dir+'rest.csv'
    focus_csv = image_dir+'focus.csv'
    batch_size = 32
    label_rest = 0
    label_focus = 1
    
    #######################################################
    ## please test if the image and label pairs are correct
    
    ######################################################
    # creat_csv(rest_csv, image_dir+'rest/', label_rest)
    # creat_csv(focus_csv, image_dir+'focus/',label_focus)
    
    rest_dataset = FacialImagesDataset(csv_file = rest_csv, root_dir = image_dir + 'rest/',
                                       transform = transforms.ToTensor())
    # print(len(rest_dataset))
    
    focus_dataset = FacialImagesDataset(csv_file = focus_csv, root_dir = image_dir + 'focus/',
                                       transform = transforms.ToTensor())
    # print(len(focus_dataset))

    img_dataset = torch.utils.data.ConcatDataset([rest_dataset, focus_dataset])
    print(len(img_dataset))
    
    ### for testing, only 338 images are used. 228 is the number of train_set, and 100 is the number of test_data. 
    train_set, test_set = torch.utils.data.random_split(img_dataset, [228,100]) 
    
    train_loader = DataLoader(dataset=train_set, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size = batch_size, shuffle=True)

 
    return train_loader, test_loader

if __name__=="__main__":
    
    train_loader, test_loader = create_datasets()
    data_iter = iter(test_loader)
    
    images, labels = next(data_iter)
    print('image size is',images.size()[0])
    print('the corresponding label are: ', labels)
    
    plt.imshow(torchvision.utils.make_grid(images, nrow=5).permute(1, 2, 0))
        
    plt.show()