#!/usr/bin/env python2
import rospy
import cv2
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
import dlib
import imutils
import numpy as np
import feature_extraction as fe
import matplotlib.pyplot as plt
import pandas as pd

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/stargazer/workspace/catkin_ws/shape_predictor_68_face_landmarks.dat')

cap=cv2.VideoCapture(0)
print(cap.isOpened())
bridge=CvBridge()
pub=rospy.Publisher('/webcam',Float32MultiArray,queue_size=1000)
rospy.init_node('image',anonymous=False)
rate=rospy.Rate(100)

homs =[]
woms = []
num_pix = []
a_r=[]
a_g=[]
a_b=[]


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640,480))
from PIL import Image


while not rospy.is_shutdown():
    ret,frame=cap.read()
    
    if(ret == True):
        image = frame#imutils.resize(frame)
        #image = cv2.flip(image, -1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        rects = detector(gray, 1)
        # loop over the face detections
        for (i, rect) in enumerate(rects):
            
            shape = predictor(gray, rect)
            shape = fe.shape_to_np(shape)

            mouth_mask = fe.mouth(shape, image)

          #  (x, y, w, h) = fe.rect_to_bb(rect)
            final = np.zeros(gray.shape)
            final[mouth_mask] = 1
            pixels = np.sum(mouth_mask)
            avg_rgb = np.mean(hsv[mouth_mask], axis = 0)

            height = fe.mouth_height(shape, image)
            width = fe.mouth_width(shape, image)

            cv2.imshow('mouth mask', final)
            # Display the landmarks
            # for i, (x, y) in enumerate(shape):
            #     # Draw the circle to mark the keypoint 
            #     cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            homs.append(height)
            woms.append(width)
            num_pix.append(pixels)
            a_r.append(avg_rgb[0])
            a_g.append(avg_rgb[1])
            a_b.append(avg_rgb[2])

        out.write(image)
        # Display the image
        im = Image.fromarray(image)
        im.save("your_file.jpeg")
        cv2.putText(img = image, text = 'HOM: {}'.format(height),org = (50,50), fontScale = 2, color = (0,0,255), fontFace = 1)
        cv2.putText(img = image, text = '# PIXELS: {}'.format(pixels),org = (50,70), fontScale = 2, color = (0,0,255), fontFace = 1)
        cv2.putText(img = image, text = 'AVG RGB: {}, {}, {}'.format(int(avg_rgb[0]),int(avg_rgb[1]), int(avg_rgb[2])),org = (50,80), fontScale = 2, color = (0,0,255), fontFace = 1)

        

        cv2.imshow('Landmark Detection', image)
       # msg3=bridge.cv2_to_imgmsg(frame,"bgr8")
        msg = Float32MultiArray()
        msg.data = shape
        # Press the escape button to terminate the code
        pub.publish(msg)
        rate.sleep()
        key=cv2.waitKey(30)
        if key==27:
            break
        if rospy.is_shutdown():
            cap.release()
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

df = pd.DataFrame(list(zip(homs, woms, num_pix, a_r, a_g, a_b)), columns = ['h', 'w', 'num', 'r', 'g','b'])
df.to_csv('data.csv')