#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import cv2
import dlib
from imutils import face_utils
import numpy as np
import time
from random import sample
from utils import filter_signal_data, get_forehead_roi,compute_bpm, plot_rgb_signals,plot_extracted_hr
import matplotlib.pyplot as plt
import scipy.io

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

mat = scipy.io.loadmat('./data/tetrisBiopac/StressTetris-1-002-1-iPhone-LightON.mat')

def find_fps(path_video):
    count=0
    video = cv2.VideoCapture(path_video +'StressTetris-1-028-1-iPhone-LightON.MOV')
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = video.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(round(fps)))  
       # Number of frames to capture
        num_frames = 100;
        print("Capturing {0} frames".format(num_frames))
        start = time.time()    
        while True :
            # print("frame {}".format(i))
            ret, frame = video.read() 
            
            count += 1
            cv2.imwrite('./data/ROIs/frame%d.jpg'% count, frame) 
            if video.get(cv2.CAP_PROP_POS_MSEC) >= 1500:
                break
        end = time.time()   
        seconds = end - start    
        print ("Time taken : {0} seconds".format(seconds))    
        fps  = num_frames / seconds    
        print("Estimated frames per second : {0}".format(fps))    
        # Release video 
        video.release()

    

def extract_frames_m1(path_video, path_image1, path_image2, video, frame_size1, frame_size2,i):
    vidcap = cv2.VideoCapture(path_video +'StressTetris-1-'+video+'-iPhone-LightON.MOV')
    success,image = vidcap.read()
    

    every_ms = 5000
    last_ts = -99999
    count = 0
    while success:    
        # Skip frames
        while vidcap.get(cv2.CAP_PROP_POS_MSEC) < every_ms + last_ts:
            # print(vidcap.get(cv2.CAP_PROP_POS_MSEC))
            if not vidcap.read()[0]:
                return
            
            
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
                image,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30),
                )

        # Crop face
        for (x, y, w, h) in faces:
            face_frame = image[y:y+h, x:x+w]
            face_frame1 = cv2.resize(face_frame, frame_size1)
            face_frame2 = cv2.resize(face_frame, frame_size2)
               
        
        if ((vidcap.get(cv2.CAP_PROP_POS_MSEC) <= 120000) or (vidcap.get(cv2.CAP_PROP_POS_MSEC) >= 300000)) and (vidcap.get(cv2.CAP_PROP_POS_MSEC) <= 540000):
            cv2.imwrite(path_image1 +"%d_%d_rest.jpg"%(i,count), face_frame1)     # save rest frame as JPEG file 
            cv2.imwrite(path_image2 +"%d_%d_rest.jpg"%(i,count), face_frame2)     # save rest frame as JPEG file 
            
            print('Read a new frame(rest): %d'%(vidcap.get(cv2.CAP_PROP_POS_MSEC)))
        elif (vidcap.get(cv2.CAP_PROP_POS_MSEC) > 120000) and (vidcap.get(cv2.CAP_PROP_POS_MSEC) < 300000):
            cv2.imwrite(path_image1 +"%d_%d_focus.jpg"%(i,count), face_frame1)   # save frame as JPEG file 
            cv2.imwrite(path_image2 +"%d_%d_focus.jpg"%(i,count), face_frame2)   # save frame as JPEG file 
            print('Read a new frame(focus): %d'%(vidcap.get(cv2.CAP_PROP_POS_MSEC)))
         
        count += 1
        last_ts = vidcap.get(cv2.CAP_PROP_POS_MSEC)
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        
def m1(path_video):
    frame_size1 = (28,28)
    frame_size2 = (224,224)
    path_img1 = './data/images_{}x{}/images/'.format(frame_size1[0],frame_size1[1])
    path_img2 = './data/images_{}x{}/images/'.format(frame_size2[0],frame_size2[1])


    if not os.path.exists(path_img1):
        os.makedirs(path_img1)
    if not os.path.exists(path_img2):
        os.makedirs(path_img2)
  
    # extract images from imgage folder
    video_nums = ['001-2','002-1','003-1','004-2','005-1','006-2','007-1','008-1','009-1','010-1'\
                  ,'011-1','013-1','015-1','017-1','018-1','019-1','021-1','022-1','023-1',\
                  '025-1', '026-1','027-1','028-1','030-1','032-1']
    for i in range(len(video_nums)):
        extract_frames_m1(path_video, path_img1, path_img2, video_nums[i], frame_size1, frame_size2, i)  
             

  
if __name__=="__main__":
    
    path_video = './data/videos/'
    
    ### m1 function extracts facial images from each frame
    m1(path_video)
