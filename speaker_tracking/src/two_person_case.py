#!/usr/bin/env python2

import pyrealsense2 as rs
import numpy as np
import cv2
import rospy
from std_msgs.msg import Float32MultiArray, Int32, Float32, Int32MultiArray
from sensor_msgs.msg import Image
from PIL import Image
from class_face import Face

from cv_bridge import CvBridge,CvBridgeError
import dlib
import imutils
import numpy as np
import feature_extraction as fe
import pandas as pd
import math 
import warnings

warnings.filterwarnings('ignore')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/stargazer/workspace/catkin_ws/shape_predictor_68_face_landmarks.dat')

pub_total =rospy.Publisher('/speaker_tracking/speech_signal',Int32MultiArray,queue_size=1000)

pub_angle =rospy.Publisher('/speaker_tracking/yaw_angle',Int32,queue_size=1000)

rospy.init_node('voice_activity',anonymous=False)
rate=rospy.Rate(120)

# Configure depth and color streams
pipeline = rs.pipeline()

config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)

if device_product_line == 'L500':
    print('t')
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    print('y')
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

# Start streaming

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()


# SET UP FEATS
total_num_pix = []
total_height = []
total_lightness = []
total_depth = []

f1 = Face()
f2 = Face()

curr_speak = 0
curr_speak_2 = 0

count = 0
flag = 0
flag_2 = 0

base_e = 0
base_h = 0
base_l = 0
base_d = 0

curr_e = 0
curr_h = 0
curr_l = 0

count = 0

disp1 = 0 
''' GET DEPTH AND GRAYSCALE IMAGES FIRST '''
try:
    while not rospy.is_shutdown():
        
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()


        align = rs.align(rs.stream.depth)
        frames_al = align.process(frames)
        depth_frame = frames_al.get_depth_frame().as_depth_frame()
        color_frame = frames_al.get_color_frame()

        # for viz
        depth_frame_v = frames.get_depth_frame()
        color_frame_v = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            #images = np.hstack((resized_color_image, depth_colormap))
            color = resized_color_image
        else:
            #images = np.hstack((color_image, depth_colormap))
            color = color_image

        #color = color_image
        depth = depth_colormap
        ''' NOW START EXTRACTING FEATURES '''

        # image = imutils.resize(color, width = int(320*1.5), height = int(240*1.5))
        # depth = imutils.resize(depth_colormap, width = int(320*1.5), height = int(240*1.5))
        # depth_raw = imutils.resize(depth_image, width = int(320*1.5), height = int(240*1.5))

        w,h = 640, 480
        image = cv2.resize(color.copy(), (w,h), interpolation = cv2.INTER_AREA)
        depth = cv2.resize(depth_colormap.copy(), (w,h), interpolation = cv2.INTER_AREA)
        depth_raw = cv2.resize(depth_image.copy(), (w,h), interpolation = cv2.INTER_AREA)


        # image = color.copy()
        # depth = depth_colormap.copy()
        # depth_raw = depth_image.copy()


        depth = fe.normalize(depth)
        depth_raw = fe.normalize(depth_raw)


        gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HLS)
        rects = detector(gray, 1)
        rect1 = 0
        rect =0

        if(len(rects) ==1):
            shape = predictor(gray, rects[0])
            f1.shape = fe.shape_to_np(shape)
            f1.image = image
            f1.depth = depth_raw
            f1.set_hsv()
            disp1 = f1.get_center_point(depth_scale)


            f1.set_mouth_mask()
            f1.current_feat = np.array([f1.get_pixels(), f1.get_mouth_height(), f1.get_lightness(), f1.get_depth(depth)]).T.reshape(4,1)
            f1.fill_features()
            f1.set_base(count)

            c1 = f1.process_features()
            c2 = 0

        elif(len(rects) == 2):
            rospy.loginfo(' 2 faces detected  ')
            shape = predictor(gray, rects[0])
            shape2 = predictor(gray, rects[1])

            f1.shape = fe.shape_to_np(shape)
            f1.image = image
            f1.depth = depth_raw
            f1.set_hsv()
            disp1 = f1.get_center_point(depth_scale)


            f2.shape = fe.shape_to_np(shape2)
            f2.image = image
            f2.depth = depth_raw
            f2.set_hsv()
            disp2 = f2.get_center_point(depth_scale)


            f1.set_mouth_mask()
            f2.set_mouth_mask()

            f1.current_feat = np.array([f1.get_pixels(), f1.get_mouth_height(), f1.get_lightness(), f1.get_depth(depth)]).T.reshape(4,1)
            f2.current_feat = np.array([f2.get_pixels(), f2.get_mouth_height(), f2.get_lightness(), f2.get_depth(depth)]).T.reshape(4,1)

            f1.fill_features()
            f2.fill_features()

            f1.set_base(count)
            f2.set_base(count)

            c1 = f1.process_features()
            c2 = f2.process_features()
          
        elif(len(rects) == 0):
            continue

        if(f1.flag == 1 or f2.flag == 1):
            msg1 = Int32MultiArray()
            msg1.data = [c1,c2]
            pub_total.publish(msg1)
     

            if(len(rects) ==1):
                y1,p1 = f1.get_yaw_pitch(disp1)
                yaw = y1


            elif(len(rects) == 2):
                y1,p1 = f1.get_yaw_pitch(disp1)
                y2,p2 = f2.get_yaw_pitch(disp2)

                yaw = (y1 + y2)/2
                #print('2 ', int(yaw))

            angle = Int32()
            angle.data = int(yaw)
            pub_angle.publish(angle)


        # t = np.zeros(image.shape)
        # t[f1.mouth_mask] =1 
        # if(len(rects) == 2):
        #     t[f2.mouth_mask] =1 

        if(len(rects)==2):
            px = (disp1[0] + disp2[0])/2
            py = (disp1[1] + disp2[1])/2

        elif(len(rects) == 1):
            px, py = disp1[0], disp1[1]


        z =np.asanyarray(color_frame_v.get_data() )
        z = cv2.resize(z, (w,h), interpolation = cv2.INTER_AREA)
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.circle(z, (w//2- int(px), h//2 - int(py)), 5, (0, 0, 255), -1)
        cv2.imshow('RealSense', z)


        cv2.waitKey(1)
        count = count + 1

finally:

    pipeline.stop()