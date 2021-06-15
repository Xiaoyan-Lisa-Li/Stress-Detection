#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 22:27:21 2021

@author: Xiaoyan
"""
import sys
import argparse
import cv2
import time

def extract_frames(pathIn, pathOut, video):
    vidcap = cv2.VideoCapture(pathIn +'StressTetris-1-'+video+'-iPhone-LightON.MOV')
    success,image = vidcap.read()
    count = 0
    start_time = time.time()
    while success:
      last_time = time.time()
      if (time.time() - start_time <= 120) or (time.time() - start_time > 300):
          cv2.imwrite(pathOut+"rest/rest_frame%d.jpg" % count, image)     # save rest frame as JPEG file 
      else:
          cv2.imwrite(pathOut+"focus/focus_frame%d.jpg" % count, image)     # save frame as JPEG file 
      count += 1
      success,image = vidcap.read()
      print('Read a new frame: ', success)
      
      skip = time.time() - last_time # how much you spent on calling functions above
      time.sleep(5 - skip) 
      
      
      
if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", default='./videos/', help="path to video")
    a.add_argument("--pathOut", default='./images/', help="path to images")
    args = a.parse_args()
    # print(args)
    video_nums = ['001-2','002-1','003-1','004-2','005-1','006-2','007-1','008-1','009-1','0010-1'\
                  ,'0011-1','0013-1','0015-1','0017-1','0018-1','0019-1','0021-1','0022-1','0023-1',\
                  '0025-1', '0026-1','0027-1','0028-1','0030-1','0032-1']
    for i in range(len(video_nums)):
        extract_frames(args.pathIn, args.pathOut,video_nums[i])      