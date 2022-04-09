#!/usr/bin/env python
# coding: utf-8

# In[7]:


#importing libraries

import numpy as np
import dlib
import cv2
import math
from scipy import ndimage    
from openni import openni2
from openni import _openni2 as c_api

#defining the dlib face detectors....
detector=dlib.get_frontal_face_detector()# detect the frontal face  
predictor=dlib.shape_predictor(r"C:\Users\CHANDRASEKAR\AppData\Local\Programs\Python\Python38\Scripts\shape_predictor_68_face_landmarks.dat")    

#initializing the pixel movement.....
face_bottom=0
face_top=0
r_face_ctr=0
p_face_ctr=0
n=0
pos=0
mir=0
depth2=0
yaw_angle=0
pitch_angle=0
#import video stream.......
cap = cv2.VideoCapture(0)

#import depth stream......
openni2.initialize()
dev = openni2.Device.open_any()
depth_stream = dev.create_depth_stream()
depth_stream.start()

depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX = 640, resolutionY = 480, fps = 30))

while True:
#get vision reading.....
    ret,frame=cap.read()
    
#convert to gray......
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
#detects number of faces.....
    dets=detector(gray,1) 
    
#get depth stream running.....
    depframe = depth_stream.read_frame()
    frame_data = depframe.get_buffer_as_uint16()
    img = np.frombuffer(frame_data, dtype=np.uint16)
    img.shape = (1, 480, 640)
    
    img = img.astype(float)#/ 1024
    img = np.concatenate((img, img, img), axis=0)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    #print("image_shape:",img.shape)
    #print("frame_shape:",frame.shape)
#looping for number of faces...
    for det in dets:
        
        shape=predictor(gray,det)  
        shape_np=np.zeros((68,2),dtype="int") #for every faces,create an array with 68 rows and 2 columns for x and y
           
        for i in range(0,68):
            if i==27 or i==8 or i==33:
                shape_np[i]=(shape.part(i).x,shape.part(i).y) #store the x and y data of all the shapes in this matrix
                
        #get face top and bottom coordinates
        face_top=shape_np[27]
        face_bottom=shape_np[8]
        
        #height of face
        px_face_ht=face_top[1]-face_bottom[1]
        mm_ref=100
        px_to_mm=mm_ref/px_face_ht
        #print(n)
        #get face ctr
        if n==0:
            p_face_ctr=shape_np[33] #previous nose(x,y)
            
        r_face_ctr=shape_np[33]
        if r_face_ctr[0]<461:
            depth=np.max(img[r_face_ctr[0]+19,r_face_ctr[1]+10,1])
        #print("Depth:",depth)
        #print("center_face:",r_face_ctr)
        #calculate x distance and y distance
        x_dist=-(r_face_ctr[0]-p_face_ctr[0])
        x_dist*=px_to_mm
        depth=depth/10
        if depth>100:
            yaw_angle=math.degrees(math.atan(x_dist/depth))
        #print("Xdist:",x_dist)
            print(depth,",",round(yaw_angle,1))

        y_dist=r_face_ctr[1]-p_face_ctr[1]
        y_dist*=px_to_mm
        
        if abs(pitch_angle)>1 : #and depth!=0:
            pitch_angle=math.degrees(math.atan(y_dist/depth))
            #print("Pitch:",pitch_angle)
        n+=1
        
        for i,(x,y) in enumerate(shape_np):
            if i==27 or i==8 or i==33:
                cv2.circle(frame,(x,y),3,(0,0,255),-1)
        img = cv2.flip(img.copy(),1)
        img=cv2.circle(img,(r_face_ctr[0]+19,r_face_ctr[1]+10),2,(0,0,255),-1)
        img=cv2.flip(img,1)
        #img=cv2.flip(cv2.circle(img,(pos[0],pos[1]),2,(0,0,255),-1),1)
    img=cv2.flip(img,1)
    #print(frame.shape)
    cv2.imshow("Depth Image", img) 
    cv2.imshow('Landmark Detection',frame)
    # Press the escape button to terminate the code
    if cv2.waitKey(30) == 27:
        break
cv2.destroyAllWindows()


# In[ ]:




