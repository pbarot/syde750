#!/usr/bin/env python2

import pyrealsense2 as rs
import numpy as np
import cv2
import rospy
from std_msgs.msg import Float32MultiArray, Int32, Float32
from sensor_msgs.msg import Image
from PIL import Image

from cv_bridge import CvBridge,CvBridgeError
import dlib
import imutils
import numpy as np
import feature_extraction as fe
import pandas as pd
import math
import warnings

warnings.filterwarnings('ignore')


class Face:
        def __init__(self):
       
            self.flag = 0

            self.features = np.zeros((4, 1))
            
        def set_mouth_mask(self):
            self.mouth_mask = fe.mouth(self.shape, self.image)
            #return self.mouth_mask

        def set_hsv(self):
        	self.hsv = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2HLS)

        def get_mouth_height(self):
            return fe.mouth_height(self.shape, self.image)

        def get_lower_face(self):
            return fe.lower_face(self.shape, self.image)

        def get_pixels(self):
            return np.sum(self.mouth_mask)/np.sum(self.get_lower_face())

        def get_lightness(self):
            return int(np.nan_to_num(np.mean(self.hsv[:,:,1][self.mouth_mask])))

        def get_depth(self, depth):
            return depth[self.mouth_mask].mean()

        def fill_features(self):
            self.features = np.concatenate((self.features, self.current_feat), axis = 1)

        def set_base(self, count):
            if(self.features.shape[1] == 5 and self.flag ==0): 
           	    self.base_vals = fe.process_features(self.features, [0,0,0,0], self.flag)
           	    self.flag = 1

            if(count%50 ==0):
                #rospy.loginfo('ESTABLISHING THRESHOLDS...')
                self.base_vals = fe.process_features(self.features, [0,0,0,0], flag = 0 )

                self.flag = 1
                self.features = np.zeros((4, 1))

        def process_features(self):  
            if(self.features.shape[1] == 10 and self.flag == 1):
                #print(self.base_vals)
                self.curr_speak = fe.process_features(self.features, self.base_vals, self.flag )
                self.features = np.zeros((4, 1))
            else:
            	self.curr_speak = 0
            return self.curr_speak

        def get_center_point(self, scale):
            idx = list(range(0,16))
            idx.extend([17,18,19,20,21,22,23,24,25,26])

            h = self.image.shape[0]//2
            w = self.image.shape[1]//2

            #x,y = (np.mean(self.shape[idx], axis = 0))
            x,y = self.shape[30]

            self.distance = self.depth[int(y),int(x)]/scale

            self.displacement = (w-x, h-y)
            return self.displacement

        def get_yaw_pitch(self, displacement):
         
            yaw = np.arctan(displacement[0]/self.distance)
            pitch = np.arctan(displacement[1]/self.distance)


            return math.degrees(yaw), math.degrees(pitch)
