### IMPORTS ###

import sys
import cv2
import time
import pickle
import numpy as np
import face_recognition

### ARGUMENTS ###

in_fname = sys.argv[1] # in video
# out_fname = sys.argv[2] # out video
# pickle_fname = sys.argv[3] # pickle file (OUT)

### MODEL - FACE ###

face_cascade = cv2.CascadeClassifier('weights/haarcascade_frontalface_default.xml')

### scaleFactor

def run_with_scaleFactor(my_scaleFactor, out_fname, pickle_fname):
    
    print('\nStarting with scaleFactor={}\n'.format(my_scaleFactor))

    ### INPUT ###

    print('Input video file: ' + in_fname)
    in_video = cv2.VideoCapture(in_fname)
    
    frame_w = int(in_video.get(3))
    frame_h = int(in_video.get(4))
    print(' - Frame width: {}, height: {}'.format(frame_w, frame_h))
    
    ### OUTPUT ###
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(out_fname, fourcc, int(in_video.get(5)), (int(in_video.get(3)), int(in_video.get(4))))
    print('Output video file: ' + out_fname)
    
    frame_counter = 0
    frame_total = in_video.get(7)
    faces_dict = {}
    
    while True:
    
        # grab single frame of video
        ret, frame = in_video.read()
        
        # if no frame, break out
        if not ret:
            break
    
        # update counter
        frame_counter += 1
        # print('{}/{}'.format(frame_counter, frame_total))
        
        start = time.time()
    
        # convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # actually detect
        faces = face_cascade.detectMultiScale(gray, my_scaleFactor, 5)
        
        # save coordinates/draw bounding boxes
        coords_list = []
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0), 2)
            coords_list.append((x,y,x+w,y+h))
        
        # display resulting image
        out_video.write(frame)
        
        end = time.time()
    
        # save info for pickle write later
        faces_dict[frame_counter] = {}
        faces_dict[frame_counter]['coords_list'] = coords_list    
        faces_dict[frame_counter]['sec'] = end-start
    
    # write to pickle file
    pickle_out = open(pickle_fname, 'wb')
    pickle.dump(faces_dict, pickle_out)
    pickle_out.close()        
    
    # release handle to the videos
    in_video.release()
    out_video.release()
    
run_with_scaleFactor(1.03, 'clip-103.avi', 'clip-103.pickle')
run_with_scaleFactor(1.05, 'clip-105.avi', 'clip-105.pickle')
run_with_scaleFactor(1.08, 'clip-108.avi', 'clip-108.pickle')
run_with_scaleFactor(1.10, 'clip-110.avi', 'clip-110.pickle')
run_with_scaleFactor(1.15, 'clip-115.avi', 'clip-115.pickle')
