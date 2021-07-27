#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from scipy import signal
import matplotlib.pyplot as plt
from scipy import fftpack
from sklearn.metrics import plot_confusion_matrix

MIN_HZ = 0.5       # 30 BPM - minimum allowed heart rate
MAX_HZ = 2.        # 120 BPM - maximum allowed heart rate

DEBUG_MODE = False
buffer_size = 500

# def plot_bmp(ecg, reconstructed_hr, path, video_nums, channel):
#     plt.figure
#     plt.plot(ecg, alpha=0.75, label='ECG')
#     plt.plot(reconstructed_hr, alpha=0.75, label='reconstructed heart rate')
#     plt.savefig(path +'StressTetris-1-{}_bmp_channel_{}.jpg'.format(video_nums, channel))
#     plt.show()

def plot_bpm(reconstructed_hr, path, video_nums, channel):
    plt.figure
    plt.plot(reconstructed_hr, alpha=0.75, label='reconstructed heart rate')
    plt.savefig(path +'StressTetris-1-{}_bmp_channel_{}.jpg'.format(video_nums, channel))
    plt.show()         
    

def plot_extracted_hr(signal_val, fps, path, video_nums, channel):

    # n = len(signal_val)  # length of the signal
    # k = np.arange(n)
    # T = n / fps
    # frq = k / T  # two sides frequency range
    # frq = frq[range(n // 2)]  # one side frequency range
    # Y = fftpack.fft(yy) / n  # fft computing and normalization
    # Y = Y[range(n // 2)] / max(Y[range(n // 2)])
    
    # plotting the data
    plt.subplot(2, 1, 1)
    plt.plot(signal_val, 'r')

    plt.xlabel('Time (micro seconds)')
    plt.ylabel('Amplitude')
    plt.grid()
    
    # # plotting the spectrum
    # plt.subplot(3, 1, 2)
    # plt.plot(frq[0:600], abs(Y[0:600]), 'k')
    # plt.xlabel('Freq (Hz)')
    # plt.ylabel('|Y(freq)|')
    # plt.grid()
    
    # plotting the specgram
    plt.subplot(2, 1, 2)
    Pxx, freqs, bins, im = plt.specgram(signal_val, NFFT=128, Fs=fps, noverlap=0)
    print("length of freqs", len(freqs))
    plt.xlabel('Time (micro seconds)')
    plt.ylabel('Frequency')
    plt.savefig(path +'StressTetris-1-{}_heart_rate_channel_{}.jpg'.format(video_nums, channel))
    plt.show()    

def plot_rgb_signals(forehead, left_cheek, right_cheek, path, video_nums):
    plt.figure
    plt.plot(forehead[:,0], '-r', alpha=0.75, label='forehead(red)')
    plt.plot(forehead[:,1], '-g', alpha=0.75, label='forehead(green)')
    plt.plot(forehead[:,2], '-b', alpha=0.75, label='forehead(blue)')

    plt.plot(left_cheek[:,0], '--r', alpha=0.75, label='left_cheek(red)')
    plt.plot(left_cheek[:,1], '--g', alpha=0.75, label='left_cheek(green)')
    plt.plot(left_cheek[:,2], '--b', alpha=0.75, label='left_cheek(blue)')

    plt.plot(right_cheek[:,0], ':r', alpha=0.75, label='right_cheek(red)')
    plt.plot(right_cheek[:,1], ':g', alpha=0.75, label='right_cheek(green)')
    plt.plot(right_cheek[:,2], ':b', alpha=0.75, label='right_cheek(blue)')
    
    plt.title('RGB values')

    plt.legend(fontsize='x-small', loc = 4)
    plt.grid(True)
    plt.savefig(path +'StressTetris-1-'+ video_nums+'_rgb.jpg')
    plt.show()
    

def compute_bpm(filtered_val, fps, time_s):
    last_bpm = 0
    bpm_ls = []
    for i in range(int(fps*time_s)):
        # Compute FFT
        if (buffer_size+i) < int(fps*time_s):
            fft = np.abs(np.fft.rfft(filtered_val[i:buffer_size+i]))
          
            # Generate list of frequencies that correspond to the FFT values
            freqs = fps / buffer_size * np.arange(buffer_size / 2 + 1) 
            # print("freqs = ",freqs)
        
            # Filter out any peaks in the FFT that are not within our range of [MIN_HZ, MAX_HZ]
            # because they correspond to impossible BPM values.
            while True:
                max_idx = fft.argmax()
                print(len(fft))
                print(max_idx)
                bps = freqs[max_idx]

                # print("bps is",bps)
                if bps < MIN_HZ or bps > MAX_HZ:
                    if DEBUG_MODE:
                        print ('BPM of {0} was discarded.'.format(bps * 60.0))
                    fft[max_idx] = 0
                else:
                    bpm = bps * 60.0
                    break 
            # It's impossible for the heart rate to change more than 10% between samples,
            # so use a weighted average to smooth the BPM with the last BPM.
            if last_bpm > 0:
                bpm = (last_bpm * 0.9) + (bpm * 0.1) 
            last_bpm = bpm
            print("bpm = ", bpm)
            bpm_ls.append(bpm)
        else:
            break
    return np.array(bpm_ls)  

def get_forehead_roi(face_points):
    # Store the points in a Numpy array so we can easily get the min and max for x and y via slicing
    points = np.zeros((len(face_points.parts()), 2))
    for i, part in enumerate(face_points.parts()):
        points[i] = (part.x, part.y)

    min_x = int(points[21, 0])
    min_y = int(min(points[21, 1], points[22, 1]))
    max_x = int(points[22, 0])
    max_y = int(max(points[21, 1], points[22, 1]))
    left = min_x
    right = max_x
    top = min_y - (max_x - min_x)
    bottom = max_y * 0.98
    return int(left), int(right), int(top), int(bottom) 

# Creates the specified Butterworth filter and applies it.
def butterworth_filter(data, low, high, sample_rate, order=5):

    nyquist_rate = sample_rate * 0.5
    # print("nyquist_rate",nyquist_rate)
    low /= nyquist_rate
    high /= nyquist_rate

    b, a = signal.butter(order, [low, high], btype='band')
    return signal.lfilter(b, a, data)

def sliding_window_demean(signal_values, num_windows):
    window_size = int(round(len(signal_values) / num_windows))
    demeaned = np.zeros(signal_values.shape)
    for i in range(0, len(signal_values), window_size):
        if i + window_size > len(signal_values):
            window_size = len(signal_values) - i
        curr_slice = signal_values[i: i + window_size]
        if DEBUG_MODE and curr_slice.size == 0:
            print ('Empty Slice: size={0}, i={1}, window_size={2}'.format(signal_values.size, i, window_size))
            print (curr_slice)
        demeaned[i:i + window_size] = curr_slice - np.mean(curr_slice)
    return demeaned

def filter_signal_data(values, fps):
    # Ensure that array doesn't have infinite or NaN values
    values = np.array(values)
    np.nan_to_num(values, copy=False)

    # Smooth the signal by detrending and demeaning
    detrended = signal.detrend(values, type='linear')
    demeaned = sliding_window_demean(detrended, 15)
    # Filter signal with Butterworth bandpass filter
    filtered = butterworth_filter(demeaned, MIN_HZ, MAX_HZ, fps, order=5)
    return filtered  

def plot_loss(model, train_loss, val_loss, results_path):
    plt.figure(figsize=(10,5))
    plt.title("Training losses")
    plt.plot(train_loss,label='train loss')
    plt.plot(val_loss, label='valid loss')
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(results_path+"{}-train-losses.png".format(model))
    plt.show()
    
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def save_objs(model, epoch, loss, optimizer, save_path):
     
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        "loss": loss,
    }, save_path)
    
def initialize_weights(model):
  if isinstance(model, nn.Conv2d):
      if model.weight is not None:
        nn.init.kaiming_uniform_(model.weight.data,nonlinearity='relu')
      if model.bias is not None:
          nn.init.constant_(model.bias.data, 0)
  elif isinstance(model, nn.BatchNorm2d):
      if model.weight is not None:
        nn.init.constant_(model.weight.data, 1)
      if model.bias is not None:  
        nn.init.constant_(model.bias.data, 0)
  elif isinstance(model, nn.Linear):
      if model.weight is not None:
        nn.init.kaiming_uniform_(model.weight.data)
      if model.bias is not None:
        nn.init.constant_(model.bias.data, 0)
        

def plot_confusion(model, x_test, y_test, class_names,results,method, n, fold):
    np.set_printoptions(precision=4)
    
    fig_path = results + '{}_figures/'.format(method)
    
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
        plt.savefig(fig_path+"{}_cofusion_matrix_{}_repeat{}_fold{}.png".format(method, normalize, n, fold))
            
        plt.show()

def plot_confusion2(y_true, y_pred, method, fig_path, fold, n, labels):
    np.set_printoptions(precision=4)
    
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['rest','focus'])

    
    # NOTE: Fill all variables here with default values of the plot_confusion_matrix
    disp = disp.plot(include_values=True,
                     cmap=plt.cm.Blues, ax=None, xticks_rotation='horizontal')
    

    plt.savefig(fig_path + "Repeat{}_Fold{}_{}_confusion_matrix.png".format(n, fold, method))
    plt.show()
        