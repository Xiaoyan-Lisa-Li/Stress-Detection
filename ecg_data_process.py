#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from utils import *
from sklearn import preprocessing

def plot_ecg(ecg, rate, seconds=540, s = 'original'):
    t = np.linspace(0, seconds, rate*seconds, endpoint=False)
    ecg = ecg[:rate*seconds]
    plt.plot(t, ecg, 'r')
    plt.title
    plt.title("{} ecg".format(s))
    plt.xlabel("time")
    plt.ylabel("mv")
    plt.show()


def plot_bpm(bpm):
    plt.figure
    plt.plot(bpm, alpha=0.75, label='reconstructed heart rate')
    plt.show()         
    
def create_ecg_data(time_s):
    origin_hz = 200
    hz = 100
    window_size = 3
    
    path_mat = './data/tetrisBiopac/'
    
    files = ['001-2','002-1','003-1','004-2','005-1','006-2','007-1','008-1','009-1','010-1'\
                  ,'011-1','013-1','015-1','016-1','017-1','018-1','019-1','021-1','022-1','023-1',\
                  '024-1','025-1', '026-1','027-1','028-1','029-1','030-1','031-1','033-1','034-1','035-1']
     
    # files = ['004-2']
    x = []
    y = []
    for i in range(len(files)):
        mat = scipy.io.loadmat(path_mat + 'StressTetris-1-'+files[i]+'-iPhone-LightON.mat')
        # print(mat)
        data = mat['data'][:,0][:time_s*origin_hz] 
        # plot_ecg(data[:600],rate=200,seconds=time_s)  
        ### downsampling from 200hz to 100hz
        sampled_data = signal.resample(data, int(round(len(data)/(origin_hz/hz))))               
        # plot_ecg(sampled_data,rate=100,seconds=time_s, s='resampled')         
        norm_data = preprocessing.normalize([sampled_data])                 
        # plot_ecg(norm_data[0],rate=100, seconds = time_s, s='resampled and normalized')
               
        filtered_data = filter_signal_data(sampled_data, fps=hz)        
        # plot_ecg(filtered_data[:300],rate=100, seconds = 3, s='resampled and filtered')

        

        data_x = norm_data.reshape(-1,window_size*hz)
        data_y = np.zeros(len(data_x))
        data_y[40:100] = 1
        
        # print(len(data_x))
        # print(data_x)
        # print(len(data_y))
        # print(data_y)
        
        x.extend(data_x)
        y.extend(data_y)
        
    return np.array(x), np.array(y)

if __name__=="__main__":
    time_s = 3
    x,y = create_ecg_data(time_s)
    # print(x)
    # print(x.shape)
    # print(y)
    # print(y.shape)