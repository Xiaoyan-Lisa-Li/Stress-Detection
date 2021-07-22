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
import numpy as np


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
    
def getint(name):
    num1, num2, _ = name.split('_')  
    return int(num1)*1000 + int(num2)

def creat_csv(csv_file, path, image_dir):
    '''
    since use model = 'a'  in "with open(csv_file, 'a', newline='')" , when do testing, 
    each time we need to create new files: focus.csv and rest.csv. 
    Another ways to save data could be explored.
    '''
  
    root = os.walk(image_dir).__next__()[0]
    print("root is",root)
    file_names = os.walk(image_dir).__next__()[2]
  
    sorted_files =sorted(file_names, key=getint)
    
    with open(path+csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image', 'label'])
        for name in sorted_files:
            # fullName = os.path.join(root, name)
            if name[-8:] == 'rest.jpg':
                writer.writerow([name, 0])
            else:
                writer.writerow([name, 1])
                    

                        
def create_datasets(batch_size, transform, image_path, image_dir, image_csv):
    
    
    img_dataset = FacialImagesDataset(csv_file = image_path+image_csv, root_dir = image_dir,
                                        transform = transform)
    

    # for i in range(len(img_dataset)):
    #     print(img_dataset[i][1])
   
    train_set, test_set = torch.utils.data.random_split(img_dataset, [round(len(img_dataset)*0.8),round(len(img_dataset)*0.2)]) 
    
    # print("train_set is ", train_set[0])
    # print("test_set is:", test_set[0])
    
    train_loader = DataLoader(dataset=train_set, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size = batch_size, shuffle=True)

 
    return train_loader, test_loader, img_dataset



if __name__=="__main__":
    batch_size = 32
    frame_size1 = (28,28)
    frame_size2 = (224,224)
    label_rest = 0
    label_focus = 1
    
    image_path1 = './data/images_{}x{}/'.format(frame_size1[0],frame_size1[1])
    image_dir1 = image_path1 + 'images/'
    image_path2 = './data/images_{}x{}/'.format(frame_size2[0],frame_size2[1])
    image_dir2 = image_path2 + 'images/'    
    img_csv = 'image.csv'

   
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])    
    
    # creat_csv(img_csv, image_path1,image_dir1)
    # creat_csv(img_csv, image_path2,image_dir2)
    
    train_loader1, test_loader1, img_dataset1 = create_datasets(batch_size, transform, image_path1, image_dir1, img_csv)
    data_iter1 = iter(train_loader1)   
    images1, labels1 = next(data_iter1)
    print('image size is',images1.size()[0])
    print('the corresponding label are: ', labels1)
    
    plt.imshow(torchvision.utils.make_grid(images1, nrow=5).permute(1, 2, 0))
    plt.show()
    
    train_loader2, test_loader2, img_dataset2 = create_datasets(batch_size, transform, image_path2, image_dir2, img_csv)
    data_iter2 = iter(train_loader2)   
    images2, labels2 = next(data_iter2)
    print('image size is',images2.size()[0])
    print('the corresponding label are: ', labels2)
    
    plt.imshow(torchvision.utils.make_grid(images2, nrow=5).permute(1, 2, 0))
        
    plt.show()
    
  
    
    
   
  
    



