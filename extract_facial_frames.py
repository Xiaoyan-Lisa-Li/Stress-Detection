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
from utils import filter_signal_data, get_forehead_roi,compute_bpm, plot_rgb_signals,plot_extracted_hr,plot_bmp
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

    

def extract_frames_m1(path_video, path_rest,path_focus, video, frame_size):
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
        
def m1(frame_size, path_video):
    
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
        
        
    # extract images from imgage folder
    video_nums = ['001-2','002-1','003-1','004-2','005-1','006-2','007-1','008-1','009-1','010-1'\
                  ,'011-1','013-1','015-1','017-1','018-1','019-1','021-1','022-1','023-1',\
                  '025-1', '026-1','027-1','028-1','030-1','032-1']
    for i in range(len(video_nums)):
        extract_frames_m1(path_video, path_rest_img, path_focus_img, video_nums[i], frame_size)  
        
       
    # ### extract images from imgage_train folder
    # video_nums = ['001-2','002-1','003-1','004-2','005-1','006-2','007-1','008-1','009-1','010-1'\
    #               ,'011-1','013-1','015-1','017-1','018-1','019-1','021-1','022-1','023-1',\
    #               '025-1', '026-1']
    # for i in range(len(video_nums)):
    #     extract_frames_m1(path_video, path_train_rest, path_train_focus, video_nums[i], frame_size)  
        
    # # extract images from imgage_test folder
    # video_nums = ['030-1','032-1']
    # for i in range(len(video_nums)):
    #     extract_frames_m1(path_video, path_test_rest, path_test_focus, video_nums[i], frame_size)  
         

def m2(path_video):
    path_ROIs = './data/ROIs/'
    detector = dlib.get_frontal_face_detector()

    # Load the predictor
    predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")
   
    if not os.path.exists(path_ROIs):
        os.makedirs(path_ROIs)
    # video_nums = ['001-2','002-1','003-1','004-2','005-1','006-2','007-1','008-1','009-1','010-1'\
    #               ,'011-1','013-1','015-1','017-1','018-1','019-1','021-1','022-1','023-1',\
    #               '025-1', '026-1','027-1','028-1','030-1','032-1']

    video_nums = ['003-1']
        
    for i in range(len(video_nums)):
        vidcap = cv2.VideoCapture(path_video +'StressTetris-1-'+ video_nums[i] +'-iPhone-LightON.MOV')
        # success,image = vidcap.read()
        count = 0
        forehead = []
        right_cheek = []
        left_cheek = []
        time_s = 539
        fps = 20
        # fps = vidcap.get(cv2.CAP_PROP_FPS)
        # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))  

        for j in range(1, time_s):  
            idx = np.arange(1,round(vidcap.get(cv2.CAP_PROP_FPS))).tolist()
            sample_id = sorted(sample(idx,20))  
            
            while True:                
                success,image = vidcap.read()
                print("current position is:",vidcap.get(cv2.CAP_PROP_POS_MSEC))              
                count += 1
                 # print('Read a new frame: ', success)  
                # Convert image into grayscale   
                
                gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                       
                # Use detector to find landmarks
                face = detector(gray) 
                                   
                if vidcap.get(cv2.CAP_PROP_POS_MSEC) >= 1000*j:   
                    idx = np.arange(1,round(vidcap.get(cv2.CAP_PROP_FPS))).tolist()
                    sample_id = sorted(sample(idx,20))
                    count = 0
                    break
                
                if len(face) == 1:
                    # Create landmark object
                    landmarks = predictor(image=gray, box=face[0])
    
                    shape = face_utils.shape_to_np(landmarks)
                    
                    if count in sample_id:
                        # left, right, top, bottom = get_forehead_roi(landmarks)
                        # f = image[top:bottom, left:right] #forehead
                        f= image[shape[76][1]:shape[21][1], shape[21][0]:shape[22][0]] #forehead
                        l = image[shape[29][1]:shape[32][1], shape[4][0]:shape[48][0]] #left cheek
                        r = image[shape[29][1]:shape[32][1], shape[54][0]:shape[26][0]] #right cheeks
                        
                        
                        forehead_rgb = f.mean(axis=(0,1))
                        right_rgb = r.mean(axis=(0,1))
                        left_rgb = l.mean(axis=(0,1))
                        
                        # print('forehead.shape',forehead_rgb.shape)
                        forehead.append(forehead_rgb)
                        right_cheek.append(right_rgb)
                        left_cheek.append(left_rgb)
                        
                        for (x, y) in shape:
                            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                        # cv2.rectangle(img=image, pt1=(left, top), pt2=(right, bottom), color=(0, 255, 0), thickness=4)
                        cv2.rectangle(img=image, pt1=(shape[21][0], shape[76][1]), pt2=(shape[22][0], shape[21][1]), color=(0, 255, 0), thickness=4)
                        cv2.rectangle(img=image, pt1=(shape[4][0], shape[29][1]), pt2=(shape[48][0], shape[32][1]), color=(0, 255, 0), thickness=4)
                        cv2.rectangle(img=image, pt1=(shape[54][0], shape[29][1]), pt2=(shape[26][0], shape[32][1]), color=(0, 255, 0), thickness=4)
                        
                        # if count == 1:
                        #     cv2.imwrite('./data/ROIs/' +"sec{}_frame{}.jpg".format(j, count), image)
     
    
        forehead = np.array(forehead)
        left_cheek = np.array(left_cheek)
        right_cheek = np.array(right_cheek)
        roi_ave = (forehead+left_cheek+right_cheek)/3
        
        print(roi_ave.shape)
        print(forehead.shape)
        print(left_cheek.shape)
        print(right_cheek.shape)
        
        plot_rgb_signals(forehead, left_cheek, right_cheek, path_ROIs, video_nums[i])        
        
        # ### extract heart rate from G signal. 
        # plot_extracted_hr(roi_ave[:,1],fps, path_ROIs, video_nums[i], channel = 'r')
        
        # ### extract heart rate from R signal. 
        # plot_extracted_hr(roi_ave[:,0],fps, path_ROIs, video_nums[i], channel = 'g')
        
        # ### extract heart rate from B signal. 
        # plot_extracted_hr(roi_ave[:,2],fps, path_ROIs, video_nums[i],  channel = 'b')
        
        # ### extract heart rate from the average of R, G, B signals
        # plot_extracted_hr(roi_ave.mean(axis=1),fps, path_ROIs, video_nums[i], channel = 'average')
        
        
        
        # Clean up the R signal data
        filtered_b = filter_signal_data(roi_ave[:,0], fps)
        # Clean up the R signal data
        filtered_g = filter_signal_data(roi_ave[:,1], fps)
        # Clean up the R signal data
        filtered_r = filter_signal_data(roi_ave[:,2], fps)  

        ### extract heart rate from G signal. 
        plot_extracted_hr(filtered_b,fps, path_ROIs, video_nums[i], channel = 'b')
        
        ### extract heart rate from R signal. 
        plot_extracted_hr(filtered_g,fps, path_ROIs, video_nums[i], channel = 'g')
        
        ### extract heart rate from B signal. 
        plot_extracted_hr(filtered_r, fps, path_ROIs, video_nums[i],  channel = 'r')
          
     
      
        out_f = path_ROIs +'StressTetris-1-'+ video_nums[i] +'-iPhone-LightON.txt'      
        roi = np.array([filtered_b,filtered_g,filtered_r])
        file = open(out_f, "w+")
        file.write(str(roi))
        file.close()
        
        
        # ### build heart rate from G signals.
        # bpm_r = compute_bpm(filtered_r, fps, time_s)
        
        ### build heart rate from G signals.
        bpm_g = compute_bpm(filtered_g, fps, time_s)
        
        # ### build heart rate from G signals.
        # bpm_b = compute_bpm(filtered_b, fps, time_s)
        
        print("bpm shpae",bpm_g.shape)

        
        # ecg = []
        
        plot_bmp(bpm_g, path_ROIs, video_nums[i], channel='g')
        plot_extracted_hr(bpm_g, fps, path_ROIs, video_nums[i], channel = 'g')
        
        
           
            
                

if __name__=="__main__":
    
    frame_size = (28, 28)   # Final frame size to save video file

    path_video = './data/videos/'
    
    # m1(frame_size,path_video)
    m2(path_video)
    # find_fps(path_video)