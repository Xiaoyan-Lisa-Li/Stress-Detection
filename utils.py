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
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.optimize import differential_evolution
from numpy import tensordot
from numpy.linalg import norm
import torchvision

MIN_HZ = 0.5       # 30 BPM - minimum allowed heart rate
MAX_HZ = 2.        # 120 BPM - maximum allowed heart rate

DEBUG_MODE = False
buffer_size = 500
def to_gpu(gpu, var):
    if gpu:
        return var.cuda()
    return var
def normalize(weights):
	# calculate l1 vector norm
	result = norm(weights, 1)
	# check for a vector of all zeros
	if result == 0.0:
		return weights
	# return normalized vector (unit norm)
	return weights / result

def evaluate_ensemble(preds, weights, y_test):
    rest_true_f = 0
    rest_false_f = 0
    focus_true_f = 0
    focus_false_f = 0
    yhats = np.array(preds)
    summed = tensordot(yhats, weights, axes=((0),(0)))
    y_ensem_f = np.argmax(summed, axis=1)
    
    
    ## calculate how many samples are predicted correctly.
    for t, e in zip(y_test, y_ensem_f):
        if t == e and t.item() == 0:
            rest_true_f += 1
        elif t != e and t.item() == 0:
            focus_false_f += 1
        elif t == e and t.item() == 1:
            focus_true_f += 1
        else:
            rest_false_f += 1
            
    acc = accuracy_score(y_test, y_ensem_f)        
    return acc, y_ensem_f, rest_true_f,focus_false_f, focus_true_f, rest_false_f

def loss_function(weights, preds, y_test):
	# normalize weights
    normalized = normalize(weights)
	### calculate error rate
    score, _, _, _, _, _ = evaluate_ensemble(preds, normalized, y_test)
    
    return 1.0 - score

def equal_wight_ensemble(preds, y_test):

    weights = [1.0/len(preds) for _ in range(len(preds))]            
    acc_ensem_f, y_ensem_f, rest_true_f,focus_false_f, focus_true_f, rest_false_f = evaluate_ensemble(preds, weights, y_test)
    print('Equal Weights Score: %.3f' % acc_ensem_f)
    return acc_ensem_f,y_ensem_f, rest_true_f,focus_false_f, focus_true_f, rest_false_f

def optimal_wight_ensemble(preds, y_test):
    # define bounds on each weight
    bound_w = [(0.0, 1.0)  for _ in range(len(preds))]
    # arguments to the loss function
    search_arg = (preds, y_test)
    # global optimization of ensemble weights
    result = differential_evolution(loss_function, bound_w, search_arg, maxiter=1000, tol=1e-7)
    # get the chosen weights
    weights = normalize(result['x'])
    print('Optimized Weights: %s' % weights)
    # evaluate chosen weights
    acc_ensem, y_ensem, rest_true,focus_false, focus_true, rest_false = evaluate_ensemble(preds, weights, y_test)
    print('Optimized Weights Score: %.3f' % acc_ensem)   
    
    return acc_ensem, y_ensem, rest_true, focus_false, focus_true, rest_false
    

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Train score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Test score")

    plt.legend(loc="best")
    return plt

def plot_curves(estimator,x,y,method, path):
    title = "Learning Curves ({})".format(method)
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    estimator = estimator
    plot_learning_curve(estimator, title, x, y, cv=cv, n_jobs=4)
    plt.savefig(path+'{}_learning_curves.jpg'.format(method))
    plt.show()
    
def plot_logloss(model, X,y, method, path):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    evalset = [(X_train, y_train), (X_test,y_test)]
    model.fit(X_train, y_train, eval_metric='logloss', eval_set=evalset)
    
# define the model
    # evaluate performance
    yhat = model.predict(X_test)
    score = accuracy_score(y_test, yhat)
    print('Accuracy: %.3f' % score)
    # retrieve performance metrics
    results = model.evals_result()
    # plot learning curves
    plt.plot(results['validation_0']['logloss'], label='train')
    plt.plot(results['validation_1']['logloss'], label='test')
    plt.title('Losses ({})'.format(method))
    plt.xlabel("Estimators")
    plt.ylabel("Log loss")
    # show the legend
    plt.legend()
    plt.grid()
    # show the plot
    plt.savefig(path+'{}_losses_curves.jpg'.format(method))
    plt.show() 

def plot_loss(model, train_loss, val_loss, results_path):
    plt.figure(figsize=(10,5))
    plt.title("Losses ({})".format(model))
    plt.plot(train_loss,label='Train loss')
    plt.plot(val_loss, label='Test loss')
    plt.xlabel("Iterations")
    plt.ylabel("Losses")
    plt.legend()
    plt.grid()
    plt.savefig(results_path+"{}-train-losses.png".format(model))
    plt.show()
    

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
    
    cm0 = confusion_matrix(y_true, y_pred, labels=labels, normalize=None)
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    
    disp0 = ConfusionMatrixDisplay(confusion_matrix=cm0, display_labels=['rest','focus'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['rest','focus'])

    
    # NOTE: Fill all variables here with default values of the plot_confusion_matrix
    disp0 = disp0.plot(include_values=True,
                     cmap=plt.cm.Blues, ax=None, xticks_rotation='horizontal')
    
    plt.savefig(fig_path + "Repeat{}_Fold{}_{}_confusion_matrix.png".format(n, fold, method))
    plt.show()
    
    disp = disp.plot(include_values=True,
                     cmap=plt.cm.Blues, ax=None, xticks_rotation='horizontal')
    

    plt.savefig(fig_path + "Repeat{}_Fold{}_{}_confusion_matrix_normalized.png".format(n, fold, method))
    plt.show()
    
def save_images(dataset, ids, path, n, clas, p):
    if len(ids) > 0:
#        samples = torch.utils.data.Subset(dataset, list(ids))
#        sample_loader = torch.utils.data.DataLoader(samples, batch_size=len(ids), shuffle=False) 
#        sample_iter = iter(sample_loader) 
#        images, labels = next(sample_iter)
#        plt.imshow(torchvision.utils.make_grid(images, nrow=5).permute(1, 2, 0))
        
        col  = 10
        rows = int(len(ids)/col)+1
        fig, axs = plt.subplots(rows,col, figsize=(5*col, 5*rows), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .1, wspace=.001)
        
        axs = axs.ravel()
        [axi.set_axis_off() for axi in axs]
        for i in range(len(ids)):    
            axs[i].imshow(dataset[ids[i]][0].permute(1, 2, 0))
            axs[i].set_title(str(ids[i]), fontdict={'fontsize': 20, 'fontweight': 'bold'})
        
        plt.savefig(path+'p{}_repeat{}_{}.png'.format(p,n,clas))
        plt.show()

def prediction_images(correct_res, incorrect_res, correct_foc, incorrect_foc, dataset, save_path, method, n):
    path0 = save_path + '{}/participant0/'.format(method)
    if not os.path.exists(path0):
        os.makedirs(path0)  

    path20 = save_path + '{}/participant20/'.format(method)
    if not os.path.exists(path20):
        os.makedirs(path20)
        
    path2 = save_path + '{}/participant2/'.format(method)
    if not os.path.exists(path2):
        os.makedirs(path2)  

    path7 = save_path + '{}/participant7/'.format(method)
    if not os.path.exists(path7):
        os.makedirs(path7)  

    path14 = save_path + '{}/participant14/'.format(method)
    if not os.path.exists(path14):
        os.makedirs(path14)
        
    path9 = save_path + '{}/participant9/'.format(method)
    if not os.path.exists(path9):
        os.makedirs(path9)
        
        
    p0 = range(0,120)
    p20 = range(2400, 2520)
    p2 = range(120, 240)
    p7 = range(840, 960)
    p14 = range(1680, 1800)
    p9 = range(1080, 1200)
    
    p0_correct_res = []
    p0_incorrect_res = []
    p0_correct_foc = []
    p0_incorrect_foc = []  
    
    p20_correct_res = []
    p20_incorrect_res = []
    p20_correct_foc = []
    p20_incorrect_foc = []
    
    p2_correct_res = []
    p2_incorrect_res = []
    p2_correct_foc = []
    p2_incorrect_foc = []
    
    p7_correct_res = []
    p7_incorrect_res = []
    p7_correct_foc = []
    p7_incorrect_foc = []
    
    p14_correct_res = []
    p14_incorrect_res = []
    p14_correct_foc = []
    p14_incorrect_foc = []

    p9_correct_res = []
    p9_incorrect_res = []
    p9_correct_foc = []
    p9_incorrect_foc = []            
    
    ###prediction of participant 0
    for i in correct_res:
        if i in p0:
            p0_correct_res.append(i)
            
    for i in incorrect_res:
        if i in p0:
            p0_incorrect_res.append(i)    
            
    for i in correct_foc:
        if i in p0:
            p0_correct_foc.append(i)
            
    for i in incorrect_foc:
        if i in p0:
            p0_incorrect_foc.append(i)   
            
    save_images(dataset, p0_correct_res, path0, n, 'correct_res', 0) 
    save_images(dataset, p0_incorrect_res, path0, n, 'incorrect_res', 0) 
    save_images(dataset, p0_correct_foc, path0, n, 'correct_foc', 0) 
    save_images(dataset, p0_incorrect_foc, path0, n, 'incorrect_foc', 0)       
    ### prediction of participant 20
    for i in correct_res:
        if i in p20:
            p20_correct_res.append(i)
            
    for i in incorrect_res:
        if i in p20:
            p20_incorrect_res.append(i)    
            
    for i in correct_foc:
        if i in p20:
            p20_correct_foc.append(i)
            
    for i in incorrect_foc:
        if i in p20:
            p20_incorrect_foc.append(i)  

    save_images(dataset, p20_correct_res, path20, n,  'correct_res', 20) 
    save_images(dataset, p20_incorrect_res, path20, n,  'incorrect_res', 20) 
    save_images(dataset, p20_correct_foc, path20, n,  'correct_foc', 20) 
    save_images(dataset, p20_incorrect_foc, path20, n,  'correct_foc', 20) 
            
    ### prediction of participant 2
    for i in correct_res:
        if i in p2:
            p2_correct_res.append(i)
            
    for i in incorrect_res:
        if i in p2:
            p2_incorrect_res.append(i)    
            
    for i in correct_foc:
        if i in p2:
            p2_correct_foc.append(i)
            
    for i in incorrect_foc:
        if i in p2:
            p2_incorrect_foc.append(i)  
            
    save_images(dataset, p2_correct_res, path2, n,  'correct_res', 2) 
    save_images(dataset, p2_incorrect_res, path2, n, 'incorrect_res', 2) 
    save_images(dataset, p2_correct_foc, path2, n,  'correct_foc', 2) 
    save_images(dataset, p2_incorrect_foc, path2, n,  'incorrect_foc', 2) 
            
    ### prediction of participant 7
    for i in correct_res:
        if i in p7:
            p7_correct_res.append(i)
            
    for i in incorrect_res:
        if i in p7:
            p7_incorrect_res.append(i)    
            
    for i in correct_foc:
        if i in p7:
            p7_correct_foc.append(i)
            
    for i in incorrect_foc:
        if i in p7:
            p7_incorrect_foc.append(i)  
            
    save_images(dataset, p7_correct_res, path7, n, 'correct_res', 7) 
    save_images(dataset, p7_incorrect_res, path7, n, 'incorrect_res', 7) 
    save_images(dataset, p7_correct_foc, path7, n, 'correct_res', 7) 
    save_images(dataset, p7_incorrect_foc, path7, n,  'incorrect_res', 7)             
    ### prediction of participant 14
    for i in correct_res:
        if i in p14:
            p14_correct_res.append(i)
            
    for i in incorrect_res:
        if i in p14:
            p14_incorrect_res.append(i)    
            
    for i in correct_foc:
        if i in p14:
            p14_correct_foc.append(i)
            
    for i in incorrect_foc:
        if i in p14:
            p14_incorrect_foc.append(i)   
            
    save_images(dataset, p14_correct_res, path14, n, 'correct_res', 14) 
    save_images(dataset, p14_incorrect_res, path14, n,  'incorrect_res', 14) 
    save_images(dataset, p14_correct_foc, path14, n,  'correct_foc', 14) 
    save_images(dataset, p14_incorrect_foc, path14, n,  'incorrect_foc', 14) 
            
    ### prediction of participant 9
    for i in correct_res:
        if i in p9:
            p9_correct_res.append(i)
            
    for i in incorrect_res:
        if i in p9:
            p9_incorrect_res.append(i)    
            
    for i in correct_foc:
        if i in p9:
            p9_correct_foc.append(i)
            
    for i in incorrect_foc:
        if i in p9:
            p9_incorrect_foc.append(i)  
            
    save_images(dataset, p9_correct_res, path9, n,  'correct_res', 9) 
    save_images(dataset, p9_incorrect_res, path9, n,  'incorrect_res', 9) 
    save_images(dataset, p9_correct_foc, path9, n,  'correct_foc', 9) 
    save_images(dataset, p9_incorrect_foc, path9, n,  'incorrect_foc', 9) 
