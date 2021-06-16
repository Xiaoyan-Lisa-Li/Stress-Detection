#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 22:27:21 2021

@author: Xiaoyan
"""
import sys
import os
import argparse
import cv2
import time

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
frame_size = (200, 200)   # Final frame size to save video file

def extract_frames(path_video, path_rest_img,path_focus_img, video):
    vidcap = cv2.VideoCapture(path_video +'StressTetris-1-'+video+'-iPhone-LightON.MOV')
    success,image = vidcap.read()
    count = 0
    start_time = time.time()
    every_ms = 5000
    last_ts = -99999

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
            face_frame = cv2.resize(face_frame, frame_size)
               
        
        if (vidcap.get(cv2.CAP_PROP_POS_MSEC) <= 120000) or (vidcap.get(cv2.CAP_PROP_POS_MSEC) > 300000):
            cv2.imwrite(path_rest_img+"rest_%s_frame%d.jpg"%(video, count), face_frame)     # save rest frame as JPEG file 
        else:
            cv2.imwrite(path_focus_img+"focus_%s_frame%d.jpg"%(video, count), face_frame)   # save frame as JPEG file 
        
         
        count += 1
        last_ts = vidcap.get(cv2.CAP_PROP_POS_MSEC)
        success,image = vidcap.read()
        print('Read a new frame: ', success)

if __name__=="__main__":

    path_video = './data/videos/'
    path_rest_img = './data/images/rest/'
    path_focus_img = './data/images/focus/'
    
    # if not os.path.exists(path_video):
    #     os.makedirs(path_video)
    if not os.path.exists(path_rest_img):
        os.makedirs(path_rest_img)
    if not os.path.exists(path_focus_img):
        os.makedirs(path_focus_img)
        
    print(args)
    video_nums = ['001-2','002-1','003-1','004-2','005-1','006-2','007-1','008-1','009-1','0010-1'\
                  ,'0011-1','0013-1','0015-1','0017-1','0018-1','0019-1','0021-1','0022-1','0023-1',\
                  '0025-1', '0026-1','0027-1','0028-1','0030-1','0032-1']
    for i in range(len(video_nums)):
        extract_frames(path_video, path_rest_img,path_focus_img, video_nums[i])  
    #below is a test line by Elizabeth    
    # extract_frames(path_video, path_rest_img,path_focus_img, '005-1')      