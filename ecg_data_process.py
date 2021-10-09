#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.io
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from scipy import signal
from utils import *
from sklearn import preprocessing

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
                    
def plot_ecg(ecg, rate, seconds=540, s = 'original'):
    t = np.linspace(0, seconds, rate*seconds, endpoint=False)
    ecg = ecg[:rate*seconds]
    fig, ax = plt.subplots()
    plt.plot(t, ecg, color='tab:blue', linewidth=2)
    plt.title
    plt.title("{}".format(s))
    plt.xlabel("time (s)")
    plt.ylabel("mv")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig("ecg_signal_{}.png".format(s))    
    plt.show()

def save_img(data,f_name):
    # s = int(len(data)/3)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(w=1,h=1)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.plot(data,linewidth=0.01)
    plt.savefig(f_name, dpi=224)
    # plt.show()
    
def plot_bpm(bpm):
    plt.figure
    plt.plot(bpm, alpha=0.75, label='reconstructed heart rate')
    plt.show()         
def create_ecg_image(time_s, window_s):
    origin_hz = 200
    new_hz = 100
    window_size = window_s
    
    path_mat = './data/ECG/'
    path_img = './data/ecg_img/images/'
    
    if not os.path.exists(path_img):
        os.makedirs(path_img)
    # 
    files = ['001-2','002-1','003-1','004-2','005-1','006-2','007-1','008-1','009-1','010-1'\
                  ,'011-1','013-1','015-1','017-1','018-1','019-1','021-1','022-1','023-1',\
                  '025-1', '026-1','027-1','028-1','030-1']
     
    # files = ['004-2']

    for i in range(len(files)):
        mat = scipy.io.loadmat(path_mat + 'StressTetris-1-'+files[i]+'-iPhone-LightON.mat')
        # print(mat)
        data = mat['data'][:,0][:time_s*origin_hz] 
        # plot_ecg(data[:600],rate=200,seconds=time_s)  
        
        ### downsampling from 200hz to 100hz
        sampled_data = signal.resample(data, int(round(len(data)/(origin_hz/new_hz))))               
        # plot_ecg(sampled_data,rate=100,seconds=time_s, s='resampled')     
        
        norm_data = preprocessing.normalize([sampled_data])[0]   
        print('norm_data',norm_data[:10])              
        # plot_ecg(norm_data[0],rate=100, seconds = time_s, s='resampled and normalized')
        count = 0
        for j in range(0, norm_data.shape[0], window_size*new_hz):
             if j >= (120*new_hz) and (j + window_size*new_hz) <= (300*new_hz):
                 data = norm_data[j:j+window_size*new_hz]
                 f_name = path_img+ "%d_%d_focus.jpg"%(i,count)
                 save_img(data,f_name)
             else:
                 data = norm_data[j:j+window_size*new_hz]
                 f_name = path_img+ "%d_%d_rest.jpg"%(i,count)
                 save_img(data,f_name)
             count+=1    
                 
        
def create_img_datasets(batch_size, transform, image_path, image_dir, image_csv):
    
    
    img_dataset = FacialImagesDataset(csv_file = image_path+image_csv, root_dir = image_dir,
                                        transform = transform)
    

    # for i in range(len(img_dataset)):
    #     print(img_dataset[i][1])
   
    train_set, test_set = torch.utils.data.random_split(img_dataset, [round(len(img_dataset)*0.8),round(len(img_dataset)*0.2)]) 
    
    # print("train_set is ", train_set[0])
    # print("test_set is:", test_set[0])
    
    train_loader = DataLoader(dataset=train_set, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size = batch_size, shuffle=False)

 
    return train_loader, test_loader, img_dataset        

def create_ecg_data(time_s, window_s):
    origin_hz = 200
    new_hz = 100
    window_size = window_s
    
    path_mat = './data/tetrisBiopac/'
    
    files = ['001-2','002-1','003-1','004-2','005-1','006-2','007-1','008-1','009-1','010-1'\
                  ,'011-1','013-1','015-1','017-1','018-1','019-1','021-1','022-1','023-1',\
                  '025-1', '026-1','027-1','028-1','030-1']

     
    x = []
    y = []
    for i in range(len(files)):
        mat = scipy.io.loadmat(path_mat + 'StressTetris-1-'+files[i]+'-iPhone-LightON.mat')
        # print(mat)
        data = mat['data'][:,0][:time_s*origin_hz] 
#        plot_ecg(data,rate=200,seconds= 70, s='ECG (200 Hz)')  
        ## downsampling from 200hz to 100hz
        sampled_data = signal.resample(data, int(round(len(data)/(origin_hz/new_hz))))               
#        plot_ecg(sampled_data,rate=100,seconds=70, s='ECG (100 Hz)')         
        norm_data = preprocessing.normalize([sampled_data])                 
#        plot_ecg(norm_data[0],rate=100, seconds = time_s, s='Normalized ECG (100 Hz)')
        ###plt.savefig('/temp/test.png', bbox_inches='tight', transparent=True, pad_inches=0)
        
        # filtered_data = filter_signal_data(sampled_data, fps=hz)        
        # plot_ecg(filtered_data[:300],rate=100, seconds = 3, s='resampled and filtered')


        data_x = norm_data.reshape(-1,window_size*new_hz)
        data_y = np.zeros(len(data_x))
        data_y[int(120/window_size):int(300/window_size)] = 1
        
        # print(len(data_x))
     
        # print(len(data_y))
       
        
        x.extend(data_x)
        y.extend(data_y)
        
    return np.array(x), np.array(y)

if __name__=="__main__":
    time_s = 360
    
    ###create ecg signal images
#    create_ecg_image(time_s,3)   
#    create_ecg_data(time_s, 3)
    
    image_path = './data/ecg_img/'
    image_dir = image_path + 'images/'    
    img_csv = 'image.csv'
    batch_size = 32
   
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])    
    
#    creat_csv(img_csv, image_path,image_dir)    
    # train_loader1, test_loader1, img_dataset1 = create_img_datasets(batch_size, transform, image_path, image_dir, img_csv)
