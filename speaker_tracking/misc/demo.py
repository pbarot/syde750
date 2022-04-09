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
import warnings

warnings.filterwarnings('ignore')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/stargazer/workspace/catkin_ws/shape_predictor_68_face_landmarks.dat')

#cap=cv2.VideoCapture("/home/stargazer/workspace/catkin_ws/speech_test_2.mp4")
#cap=cv2.VideoCapture(6)

#print(cap.isOpened())
# bridge=CvBridge()
pub =rospy.Publisher('/speaker_tracking/speech_signal',Int32,queue_size=1000)
pub_angle =rospy.Publisher('/speaker_tracking/bearing',Int32,queue_size=1000)

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


massive = np.zeros((4,1))
features = np.zeros((4, 1))


curr_speak = 0

count = 0
flag = 0

base_e = 0
base_h = 0
base_l = 0
base_d = 0

curr_e = 0
curr_h = 0
curr_l = 0

count = 0
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

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
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
        image = color.copy()#cv2.resize(color.copy(), (w,h), interpolation = cv2.INTER_AREA)
        depth = depth_colormap.copy()#cv2.resize(depth_colormap.copy(), (w,h), interpolation = cv2.INTER_AREA)
        depth_raw = depth_image.copy()#cv2.resize(depth_image.copy(), (w,h), interpolation = cv2.INTER_AREA)


        # image = color.copy()
        # depth = depth_colormap.copy()
        # depth_raw = depth_image.copy()


        # depth = fe.normalize(depth)
        # depth_raw = fe.normalize(depth_raw)


        gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HLS)
        rects = detector(gray, 1)

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            
            shape = predictor(gray, rect)
            shape = fe.shape_to_np(shape)

            mouth_mask = fe.mouth(shape, image)
            # depth[mouth_mask] = 0
            # image[mouth_mask] = 0
            # get features
            height = fe.mouth_height(shape, image)/fe.full_height(shape, image)
            lf = fe.lower_face(shape, image)

            # i = np.zeros((image.shape[0], image.shape[1]))
            # i[lf] = 1

            pixels = np.sum(mouth_mask)/np.sum(lf)
            lightness = np.nan_to_num(np.mean(hsv[:,:,1][mouth_mask]))#/np.mean(hsv[:,:,1][lf]))
            depth_avg = depth[mouth_mask].mean()

            current_feat = np.array([pixels, height, int(lightness), depth_avg]).T  
            current_feat = current_feat.reshape(4,1)

            massive = np.concatenate((massive, current_feat), axis = 1)

            t = np.zeros(image.shape)
            t[lf] =1 

            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            idx = list(range(0,16))
            idx.extend([17,18,19,20,21,22,23,24,25,26])

            h = image.shape[0]//2
            w = image.shape[1]//2

            x,y = np.mean(shape[idx], axis = 0)
            #print(x,y)
            z = image
            cv2.circle(z, (int(x),int(y)), 1, (0, 0, 255), -1)
       # cv2.imshow('RealSense', np.asanyarray(color_frame_v.get_data()))
            # cv2.putText(img = x, text = 'lightness: {}'.format(round(lightness,2)),org = (50,50), fontScale = 2, color = (0,0,255), fontFace = 1)
            # cv2.putText(img = x, text = 'depth: {}'.format(round(depth_avg, 2)),org = (50,70), fontScale = 2, color = (0,0,255), fontFace = 1)

            cv2.imshow('RealSense',z)

            cv2.waitKey(1)
        count = count + 1

    #np.save("/home/stargazer/workspace/catkin_ws/src/speaker_tracking/src/fulldata.npy", massive)
finally:

    # Stop streaming
    pipeline.stop()