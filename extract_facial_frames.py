#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import cv2
import time

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
frame_size = (28, 28)   # Final frame size to save video file

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
               
        
        if ((vidcap.get(cv2.CAP_PROP_POS_MSEC) <= 120000) or (vidcap.get(cv2.CAP_PROP_POS_MSEC) >= 300000)) and (vidcap.get(cv2.CAP_PROP_POS_MSEC) <= 540000):
            cv2.imwrite(path_rest_img+"rest_%s_frame%d.jpg"%(video, count), face_frame)     # save rest frame as JPEG file 
            print('Read a new frame(rest): %d'%(vidcap.get(cv2.CAP_PROP_POS_MSEC)))
        elif (vidcap.get(cv2.CAP_PROP_POS_MSEC) > 120000) and (vidcap.get(cv2.CAP_PROP_POS_MSEC) < 300000):
            cv2.imwrite(path_focus_img+"focus_%s_frame%d.jpg"%(video, count), face_frame)   # save frame as JPEG file 
            print('Read a new frame(focus): %d'%(vidcap.get(cv2.CAP_PROP_POS_MSEC)))
         
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
        
    # print(args)
    video_nums = ['001-2','002-1','003-1','004-2','005-1','006-2','007-1','008-1','009-1','010-1'\
                  ,'011-1','013-1','015-1','017-1','018-1','019-1','021-1','022-1','023-1',\
                  '025-1', '026-1','027-1','028-1','030-1','032-1']
    for i in range(len(video_nums)):
        extract_frames(path_video, path_rest_img,path_focus_img, video_nums[i])  
 
