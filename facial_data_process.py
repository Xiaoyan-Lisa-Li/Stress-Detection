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
                    

                        
def create_datasets(batch_size, transform, image_dir, rest_csv, focus_csv):
    
    
    rest_data = FacialImagesDataset(csv_file = image_dir+rest_csv, root_dir = image_dir + 'rest/',
                                        transform = transform)
    
    
    focus_data = FacialImagesDataset(csv_file = image_dir+focus_csv, root_dir = image_dir + 'focus/',
                                        transform = transform)
   
    
    img_dataset = torch.utils.data.ConcatDataset([rest_data, focus_data])

    
    # for i in range(len(img_dataset)):
    #     print(img_dataset[i][1])
   
    train_set, test_set = torch.utils.data.random_split(img_dataset, [round(len(img_dataset)*0.8),round(len(img_dataset)*0.2)]) 
    
    # print("train_set is ", train_set[0])
    # print("test_set is:", test_set[0])
    
    train_loader = DataLoader(dataset=train_set, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size = batch_size, shuffle=True)

 
    return train_loader, test_loader, img_dataset

def create_datasets2(batch_size, transform, image_train_dir, image_test_dir, rest_csv, focus_csv):
    
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
    img_dataset = torch.utils.data.ConcatDataset([train_set, test_set])
 
    return train_loader, test_loader, img_dataset



if __name__=="__main__":
    batch_size = 32
    frame_size = (28,28)
    label_rest = 0
    label_focus = 1
    
    image_dir = './data/images_{}x{}/'.format(frame_size[0],frame_size[1])
    image_train_dir = './data/data2/images_train_{}x{}/'.format(frame_size[0],frame_size[1])
    image_test_dir = './data/data2/images_test_{}x{}/'.format(frame_size[0],frame_size[1])
    
    rest_csv = 'rest.csv'
    focus_csv = 'focus.csv'

    
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    
    ### both trian and test data are from all 25 videos or       
    creat_csv(image_dir + rest_csv, image_dir +'rest/', label_rest)
    creat_csv(image_dir + focus_csv, image_dir +'focus/',label_focus)
    train_loader, test_loader, img_dataset = create_datasets(batch_size,transform, image_dir, rest_csv, focus_csv)
    
    
    # # or trian data is from first 24 videos and test data is from last 1 videos.
    # creat_csv(image_train_dir + rest_csv, image_train_dir +'rest/', label_rest)
    # creat_csv(image_train_dir + focus_csv, image_train_dir +'focus/',label_focus)       
    # creat_csv(image_test_dir + rest_csv, image_test_dir +'rest/', label_rest)
    # creat_csv(image_test_dir + focus_csv, image_test_dir +'focus/',label_focus)  
    # train_loader, test_loader, img_dataset = create_datasets2(batch_size,transform, image_train_dir, image_test_dir, rest_csv, focus_csv)
    
    
    data_iter = iter(train_loader)   
    images, labels = next(data_iter)
    print('image size is',images.size()[0])
    print('the corresponding label are: ', labels)
    
    plt.imshow(torchvision.utils.make_grid(images, nrow=5).permute(1, 2, 0))
        
    plt.show()
    
  
    
    
   
  
    



