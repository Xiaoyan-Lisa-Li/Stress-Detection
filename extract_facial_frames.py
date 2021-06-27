#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import cv2


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def extract_frames(path_video, path_rest,path_focus, video, frame_size):
    vidcap = cv2.VideoCapture(path_video +'StressTetris-1-'+video+'-iPhone-LightON.MOV')
    success,image = vidcap.read()
    count = 0

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
            cv2.imwrite(path_rest +"rest_%s_frame%d.jpg"%(video, count), face_frame)     # save rest frame as JPEG file 
            print('Read a new frame(rest): %d'%(vidcap.get(cv2.CAP_PROP_POS_MSEC)))
        elif (vidcap.get(cv2.CAP_PROP_POS_MSEC) > 120000) and (vidcap.get(cv2.CAP_PROP_POS_MSEC) < 300000):
            cv2.imwrite(path_focus +"focus_%s_frame%d.jpg"%(video, count), face_frame)   # save frame as JPEG file 
            print('Read a new frame(focus): %d'%(vidcap.get(cv2.CAP_PROP_POS_MSEC)))
         
        count += 1
        last_ts = vidcap.get(cv2.CAP_PROP_POS_MSEC)
        success,image = vidcap.read()
        print('Read a new frame: ', success)

if __name__=="__main__":
    
    frame_size = (224, 224)   # Final frame size to save video file

    path_video = './data/videos/'
    
    path_rest_img = './data/images_{}x{}/rest/'.format(frame_size[0],frame_size[1])
    path_focus_img = './data/images_{}x{}/focus/'.format(frame_size[0],frame_size[1])
    
    path_train_rest = './data/images_train_{}x{}/rest/'.format(frame_size[0],frame_size[1])
    path_train_focus = './data/images_train_{}x{}/focus/'.format(frame_size[0],frame_size[1])
    
    path_test_rest = './data/images_test_{}x{}/rest/'.format(frame_size[0],frame_size[1])
    path_test_focus = './data/images_test_{}x{}/focus/'.format(frame_size[0],frame_size[1])
    

    if not os.path.exists(path_rest_img):
        os.makedirs(path_rest_img)
    if not os.path.exists(path_focus_img):
        os.makedirs(path_focus_img)
        
    if not os.path.exists(path_train_rest):
        os.makedirs(path_train_rest)
    if not os.path.exists(path_train_focus):
        os.makedirs(path_train_focus)

    if not os.path.exists(path_test_rest):
        os.makedirs(path_test_rest)
    if not os.path.exists(path_test_focus):
        os.makedirs(path_test_focus)        
        
        
    # # extract images from imgage folder
    # video_nums = ['001-2','002-1','003-1','004-2','005-1','006-2','007-1','008-1','009-1','010-1'\
    #               ,'011-1','013-1','015-1','017-1','018-1','019-1','021-1','022-1','023-1',\
    #               '025-1', '026-1','027-1','028-1','030-1','032-1']
    # for i in range(len(video_nums)):
    #     extract_frames(path_video, path_rest_img, path_focus_img, video_nums[i], frame_size)  
        
       
    # ### extract images from imgage_train folder
    # video_nums = ['001-2','002-1','003-1','004-2','005-1','006-2','007-1','008-1','009-1','010-1'\
    #               ,'011-1','013-1','015-1','017-1','018-1','019-1','021-1','022-1','023-1',\
    #               '025-1', '026-1']
    # for i in range(len(video_nums)):
    #     extract_frames(path_video, path_train_rest, path_train_focus, video_nums[i], frame_size)  
        
    # extract images from imgage folder
    video_nums = ['027-1','028-1','030-1','032-1']
    for i in range(len(video_nums)):
        extract_frames(path_video, path_test_rest, path_test_focus, video_nums[i], frame_size)  
        