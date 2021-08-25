#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from venn import venn
from matplotlib_venn import venn2, venn3
from matplotlib import pyplot as plt
import pickle
import numpy as np
import os

vgg_pred_f = np.load('./weighted_ensemble_results_6mins/facial_image/Pretrained_VGG_face/pred_pkl/Pretrained_VGG_face_repeat2_fold0_pred_index2.pkl', allow_pickle=True)
svm_pred_f = np.load('./weighted_ensemble_results_6mins/facial_image/SVM_face/pred_pkl/SVM_face_repeat2_fold0_pred_index2.pkl', allow_pickle=True)
xgboost_pred_f = np.load('./weighted_ensemble_results_6mins/facial_image/XGBoost_face/pred_pkl/XGBoost_face_repeat2_fold0_pred_index2.pkl',allow_pickle=True)
true_f = np.load('./weighted_ensemble_results_6mins/facial_image/SVM_face/pred_pkl/SVM_face_repeat2_fold0_true_index.pkl', allow_pickle=True)
svm_pred_f = np.array(svm_pred_f)
vgg_pred_f = np.array(vgg_pred_f)
xgboost_pred_f = np.array(xgboost_pred_f)


cnn_pred_e = np.load('./weighted_ensemble_results_6mins/ecg/1D_CNN_ECG/pred_pkl/1D_CNN_ECG_repeat2_fold0_pred_index2.pkl', allow_pickle=True)
vgg_pred_e = np.load('./weighted_ensemble_results_6mins/ecg/Pretrained_VGG_ECG/pred_pkl/Pretrained_VGG_ECG_repeat2_fold0_pred_index2.pkl', allow_pickle=True)
svm_pred_e = np.load('./weighted_ensemble_results_6mins/ecg/SVM_ECG/pred_pkl/SVM_ECG_repeat2_fold0_pred_index2.pkl', allow_pickle=True)
xgboost_pred_e = np.load('./weighted_ensemble_results_6mins/ecg/XGBoost_ECG/pred_pkl/XGBoost_ECG_repeat2_fold0_pred_index2.pkl',allow_pickle=True)
true_e = np.load('./weighted_ensemble_results_6mins/ecg/SVM_ECG/pred_pkl/SVM_ECG_repeat2_fold0_true_index.pkl', allow_pickle=True)
cnn_pred_e = np.array(cnn_pred_e)
vgg_pred_e = np.array(vgg_pred_e)
svm_pred_e = np.array(svm_pred_e)
xgboost_pred_e = np.array(xgboost_pred_e)
true_e = np.array(true_e)

rest_num = len(true_f[0])
focus_num = len(true_f[1])
print('rest_num=',rest_num)
print('focus_num=',focus_num)
##
#print('svm_pred_e_rest = ',svm_pred_e[0])
#print('svm_pred_e_rest = ',len(svm_pred_e[0]))
#print('svm_pred_e_focus = ',svm_pred_e[1])
#print('svm_pred_e_focus = ',len(svm_pred_e[1]))

#print('xgboost_pred_e_rest = ',xgboost_pred_e[0])
#print('xgboost_pred_e_rest = ',len(xgboost_pred_e[0]))
#print('xgboost_pred_e_focus = ',xgboost_pred_e[1])
#print('xgboost_pred_e_focus = ',len(xgboost_pred_e[1]))

#print('vgg_pred_f = ',vgg_pred_f[0])
#print('xgboost_pred_e = ',len(xgboost_pred_e[0]))
print('true_e = ',true_e)
    
def plot_venn(preds, title, name):
    venn(preds)
    plt.title(title)
    plt.savefig(name)
    plt.show()

    
    
if __name__=="__main__":   
    path = './venn_diagram/'
    if not os.path.exists(path):
        os.makedirs(path)
    ###########################################################################
    ### rest venn diagram
    face_rest_t = 'Rest sets (facial images)' 
    preds_f = {
    "VGG facial images": set(vgg_pred_f[0]),
    "SVM facial images": set(svm_pred_f[0]),
    "XGBoost facial images": set(xgboost_pred_f[0]),
    'Target ({} samples)'.format(rest_num): set(true_f[0])}
    preds_f2 = {
    "VGG facial images": set(vgg_pred_f[0]),
    "SVM facial images": set(svm_pred_f[0]),
    "XGBoost facial images": set(xgboost_pred_f[0])}
    
    plot_venn(preds_f, face_rest_t, path+'rest_face_4sets.png')
    plot_venn(preds_f2, face_rest_t, path+'rest_face_3sets.png')  
    
    preds_e = {
    "VGG ECG": set(vgg_pred_e[0]),
    "SVM ECG": set(svm_pred_e[0]),
    "XGBoost ECG": set(xgboost_pred_e[0]),
    '1D CNN ECG': set(cnn_pred_e[0]),
    'Target ({} samples)'.format(rest_num): set(true_f[0])}   
    ecg_rest_t = 'Rest sets (ECG)' 
    plot_venn(preds_e, ecg_rest_t, path+'rest_ecg.png')
    
    preds_svm={
    "SVM facial images": set(svm_pred_f[0]),
    "SVM ECG": set(svm_pred_e[0]),
    'Target ({} samples)'.format(rest_num): set(true_f[0])}
    svm_rest_t = 'Rest sets (SVMs)'
    plot_venn(preds_svm, svm_rest_t, path+'rest_svm.png')
    
    preds_xgb={
    "XGBoost facial images": set(xgboost_pred_f[0]),
    "XGBoost ECG": set(xgboost_pred_e[0]),
    'Target ({} samples)'.format(rest_num): set(true_f[0])}
    xgboost_rest_t = 'Rest sets (XGBoosts)'
    plot_venn(preds_xgb, xgboost_rest_t, path+'rest_xgboost.png')
    
    preds_vgg_cnn={
    "VGG facial images": set(vgg_pred_f[0]),
    "VGG ECG": set(vgg_pred_e[0]),
    'Target ({} samples)'.format(rest_num): set(true_f[0])}
    vgg_cnn = 'Rest sets (VGGs)'
    plot_venn(preds_vgg_cnn, vgg_cnn, path+'rest_vgg.png')
    
    ###########################################################################
    ### focus venn diagram
    face_rest_t = 'Focus sets (facial images)' 
    preds_f = {
    "VGG facial images": set(vgg_pred_f[1]),
    "SVM facial images": set(svm_pred_f[1]),
    "XGBoost facial images": set(xgboost_pred_f[1]),
    'Target ({} samples)'.format(focus_num): set(true_f[1])}
    preds_f2 = {
    "VGG facial images": set(vgg_pred_f[1]),
    "SVM facial images": set(svm_pred_f[1]),
    "XGBoost facial images": set(xgboost_pred_f[1])}
    
    plot_venn(preds_f, face_rest_t, path+'focus_face_4sets.png')
    plot_venn(preds_f2, face_rest_t, path+'focus_face_3sets.png')  
    
    preds_e = {
    "VGG ECG": set(vgg_pred_e[1]),
    "SVM ECG": set(svm_pred_e[1]),
    "XGBoost ECG": set(xgboost_pred_e[1]),
    '1D CNN ECG': set(cnn_pred_e[1]),
    'Target ({} samples)'.format(focus_num): set(true_f[1])}   
    ecg_rest_t = 'Focus sets (ECG)' 
    plot_venn(preds_e, ecg_rest_t, path+'focus_ecg.png')
    
    preds_svm={
            "SVM facial images": set(svm_pred_f[1]),
            "SVM ECG": set(svm_pred_e[1]),
            'Target ({} samples)'.format(focus_num): set(true_f[1])}
    svm_rest_t = 'Focus sets (SVMs)'
    plot_venn(preds_svm, svm_rest_t, path+'focus_svm.png')
    
    preds_xgb={
            "XGBoost facial images": set(xgboost_pred_f[1]),
            "XGBoost ECG": set(xgboost_pred_e[1]),
            'Target ({} samples)'.format(focus_num): set(true_f[1])}
    xgboost_rest_t = 'Focus sets (XGBoosts)'
    plot_venn(preds_xgb, xgboost_rest_t, path+'focus_xgboost.png')
    
    preds_vgg_cnn={
            "VGG facial images": set(vgg_pred_f[1]),
            "VGG ECG": set(vgg_pred_e[1]),
            'Target ({} samples)'.format(focus_num): set(true_f[1])}
    vgg_cnn = 'Focus sets (VGGs)'
    plot_venn(preds_vgg_cnn, vgg_cnn, path+'focus_vgg.png')    
