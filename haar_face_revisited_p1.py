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

### MODEL - BODY ###

# body_cascade = cv2.CascadeClassifier('../weights/haarcascade_fullbody.xml')

### MODEL - FACE ###

face_cascade_def = cv2.CascadeClassifier('../weights/haarcascade_frontalface_default.xml')
face_cascade_alt = cv2.CascadeClassifier('../weights/haarcascade_frontalface_alt.xml')
face_cascade_alt2 = cv2.CascadeClassifier('../weights/haarcascade_frontalface_alt2.xml')
face_cascade_tree = cv2.CascadeClassifier('../weights/haarcascade_frontalface_alt_tree.xml')

### INPUT ###

def run_with_cascade(face_cascade, out_fname, pickle_fname):

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
        print('{}/{}'.format(frame_counter, frame_total))
        
        start = time.time()
    
        # convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # actually detect
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
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
    
run_with_cascade(face_cascade_def, 'Station1-def.avi', 'Station1-def.pickle')
run_with_cascade(face_cascade_alt, 'Station1-alt.avi', 'Station1-alt.pickle')
run_with_cascade(face_cascade_alt2, 'Station1-alt2.avi', 'Station1-alt2.pickle')
run_with_cascade(face_cascade_tree, 'Station1-tree.avi', 'Station1-tree.pickle')

