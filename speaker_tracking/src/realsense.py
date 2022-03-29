
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
import matplotlib.pyplot as plt
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
pub=rospy.Publisher('/speaker_tracking/speech_signal',Int32,queue_size=1000)
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

pipeline.start(config)


# SET UP FEATS
total_num_pix = []
total_height = []
total_lightness = []
total_depth = []

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

        # depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()
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
        # if depth_colormap_dim != color_colormap_dim:
        #     resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        #     #images = np.hstack((resized_color_image, depth_colormap))
        #     color = resized_color_image
        # else:
        #     #images = np.hstack((color_image, depth_colormap))
        #     color = color_image

        color = color_image
        depth = depth_colormap
        ''' NOW START EXTRACTING FEATURES '''

        image = imutils.resize(color, width = int(320), height = int(240))
        depth = imutils.resize(depth, width = int(320), height = int(240))

        depth = fe.normalize(depth)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
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
            depth_avg =depth[mouth_mask].mean()

            # total_num_pix.append(pixels)
            # total_height.append(height)
            # total_lightness.append(int(lightness))
            # total_depth.append(depth_avg)

            current_feat = np.array([pixels, height, int(lightness), depth_avg]).T  
            current_feat = current_feat.reshape(4,1)
            features = np.concatenate((features, current_feat), axis = 1)


            if(features.shape[1] == 5 and flag ==0):
                # base_e = fe.process_npix(total_num_pix, base_e, flag)
                # base_h = fe.process_heights(total_height, base_h, flag)
                # base_l = fe.process_lightness(total_lightness, base_l, flag)

                base_vals = fe.process_features(features, [base_e, base_h, base_l,base_d], flag )
                print(base_vals)

                energy = 0
                flag = 1
                features = np.zeros((4, 1))

                # total_num_pix = []
                # total_height = []
                # total_lightness = []


            elif(features.shape[1] == 10 and flag == 1):
                # curr_e, energy_p = fe.process_npix(total_num_pix,base_e,  flag)
                # curr_h, energy_h = fe.process_heights(total_height,base_h,  flag)
                # curr_l, energy_l = fe.process_lightness(total_lightness, base_l, flag)

                curr_speak = fe.process_features(features, base_vals, flag )

                #curr_speak.append(np.bitwise_and(np.bitwise_and(curr_e, curr_h), curr_l))
                #print(np.bitwise_and(np.bitwise_and(curr_e, curr_h), curr_l))

                features = np.zeros((4, 1))
              #  print(curr_speak[3], curr_speak[4])
                print(curr_speak)
                # total_num_pix = []
                # total_height = []
                # total_lightness = []
                # print(curr_h)

        if(flag == 1):
            msg = Int32()
            if(curr_e !=None):
                msg.data = curr_speak#np.bitwise_and(curr_h, curr_l)# np.bitwise_and(np.bitwise_and(curr_h, curr_l), curr_l)
                pub.publish(msg)

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', depth)#np.hstack((image, depth)))
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()