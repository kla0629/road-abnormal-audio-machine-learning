import csv
from collections import defaultdict
import os
import sys

import numpy as np
import tensorflow as tf

import evaluation
import inputs
import model

import wave
from scipy.io.wavfile import write
from scipy.io import wavfile
import time

import sed_vis
import dcase_util
import pyaudio 
import wave
import threading


import cv2
import pygame

global Time_check


def media_run():
    capture = cv2.VideoCapture("The Fast and the Furious.mp4")

    pygame.mixer.init() 

    soundfile = "The Fast and the Furious.wav" 
    sound = pygame.mixer.Sound(soundfile) 
    sound.play()

    while True:
        if(capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
            capture.open("The Fast and the Furious.mp4")
        
        ret, frame = capture.read()
        cv2.imshow("VideoFrame", frame)
    

        if cv2.waitKey(30) > 0 : #frame time set
            break

    capture.release()
    cv2.destroyAllWindows()


def Check_time():
    global Time_check
    while True:
        Time_check = True
        time.sleep(0.9)
        Time_check = False
        time.sleep(0.01)


t1 = threading.Thread(target = media_run)
t2 = threading.Thread(target = Check_time)

with open("Total.txt", "r") as f:
    all_line = f.readlines()

flag = 1

soundfile = "The Fast and the Furious.wav" 
samplingFrequency, signalData = wavfile.read(soundfile)


line = all_line[0].split()
fir_txt = line[0]
count = 1

t1.start()
t2.start()

while True:
    try:
        if flag is 1:
            if count < 180:
                data = signalData[:count*samplingFrequency]
                with open("tests/data/1.txt",'w') as f:
                    for i in range(0,count):
                        f.write(all_line[i])


            else:
                data = signalData[(count-180)*samplingFrequency : count*samplingFrequency]

                with open("tests/data/1.txt",'w') as f:
                    for i in range(count-180,count):
                        line = all_line[i].split()
                        f.write(fir_txt + "\t" + str(i + 180-count) + '\t' + str(i + 180 - count +1) +'\t' + line[3]+'\n')


        elif flag is -1:
            if count < 180:
                data = signalData[:count*samplingFrequency]
                with open("tests/data/2.txt",'w') as f:
                   for i in range(0,count):
                       f.write(all_line[i])
            else:
               data = signalData[(count-180)*samplingFrequency : count*samplingFrequency]
               with open("tests/data/2.txt",'w') as f:
                    for i in range(count-180,count):
                        line = all_line[i].split()
                        f.write(fir_txt + "\t" + str(i + 180-count) + '\t' + str(i + 180 - count +1) +'\t' + line[3]+'\n')

        write("tests/data/a001.wav",samplingFrequency,data)

        audio_container = dcase_util.containers.AudioContainer().load(
        'tests/data/a001.wav'
        )
        if flag is 1:
            os.remove('tests/data/1.ann')
            os.rename('tests/data/1.txt','tests/data/1.ann')
            ann_name = 'tests/data/1.ann'
        else:
            os.remove('tests/data/2.ann')
            os.rename('tests/data/2.txt','tests/data/2.ann')
            ann_name = 'tests/data/2.ann'
        estimated_event_list = dcase_util.containers.MetaDataContainer().load(
            ann_name
        )

        event_lists = {
        'estimated': estimated_event_list
        }

        vis = sed_vis.visualization.EventListVisualizer(event_lists=event_lists,
                                                    audio_signal=audio_container.data,
                                                    sampling_rate=audio_container.fs)
                            
    
        vis.show()

        
        count += 1
        flag *=-1
        
        while Time_check:
            disgard = 1
        print("break")

                
    except:
        break