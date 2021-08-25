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

    

def extract_frames(path_video, path_image, video, frame_size,i):
    vidcap = cv2.VideoCapture(path_video +'StressTetris-1-'+video+'-iPhone-LightON.MOV')
    success,image = vidcap.read()
    
    x_P = 0
    y_p = 0
    every_ms = 3000
    last_ts = 0
    count = 0
    while success:             
            
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
                image,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(30, 30),
                )

        # Crop face
        for (x, y, w, h) in faces:
            face_frame = image[y:y+h, x:x+w]
            # if count != 0 and abs(y-y_p) > h/2:
            print('number of faces is',len(faces))
           
            if count != 0 and abs(y-y_p) > h/2:
                if len(faces)>1:
                    print('continue')
                    continue
                else:
                    face_frame2 = image
                    break
            print("count = ",count)
            # print('x=',x)
            # print("y=",y)
            # print('w=',w)
            # print("h=",h) 
            face_frame2 = cv2.resize(face_frame, frame_size)
            x_p = x
            y_p = y
               
   
        if ((vidcap.get(cv2.CAP_PROP_POS_MSEC) < 120000) or (vidcap.get(cv2.CAP_PROP_POS_MSEC) >= 300000)) and (vidcap.get(cv2.CAP_PROP_POS_MSEC) < 360000):
            cv2.imwrite(path_image +"%d_%d_rest.jpg"%(i,count), face_frame2)     # save rest frame as JPEG file 
            print('Read a new frame(rest): %d'%(vidcap.get(cv2.CAP_PROP_POS_MSEC)))
        elif (vidcap.get(cv2.CAP_PROP_POS_MSEC) >= 120000) and (vidcap.get(cv2.CAP_PROP_POS_MSEC) < 300000):
            cv2.imwrite(path_image +"%d_%d_focus.jpg"%(i,count), face_frame2)   # save frame as JPEG file 
            print('Read a new frame(focus): %d'%(vidcap.get(cv2.CAP_PROP_POS_MSEC)))
         
        # Skip frames
        while vidcap.get(cv2.CAP_PROP_POS_MSEC) < every_ms + last_ts:
            # print(vidcap.get(cv2.CAP_PROP_POS_MSEC))
            if not vidcap.read()[0]:
                break
             
        count += 1

        last_ts = vidcap.get(cv2.CAP_PROP_POS_MSEC)
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        
def main(path_video):
    frame_size = (224,224)
    path_img = './data/images_{}x{}/images/'.format(frame_size[0],frame_size[1])


    if not os.path.exists(path_img):
        os.makedirs(path_img)
  
    ###extract images from imgage folder
    video_nums = ['001-2','002-1','003-1','004-2','005-1','006-2','007-1','008-1','009-1','010-1'\
                  ,'011-1','013-1','015-1','017-1','018-1','019-1','021-1','022-1','023-1',\
                  '025-1', '026-1','027-1','028-1','030-1']
    
   
    for i in range(len(video_nums)):
        extract_frames(path_video, path_img, video_nums[i], frame_size, i)  
             

  
if __name__=="__main__":
    
    path_video = './data/videos/'
    
    ### m1 function extracts facial images from each frame
    main(path_video)
