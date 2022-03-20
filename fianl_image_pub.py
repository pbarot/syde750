#!/usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
import dlib
import numpy as np
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/opt/pal/ferrum/shape_predictor_68_face_landmarks.dat')
cap=cv2.VideoCapture(0)
print(cap.isOpened())
bridge=CvBridge()
pub=rospy.Publisher('/webcam',Image,queue_size=1000)
rospy.init_node('image',anonymous=False)
rate=rospy.Rate(30)
while not rospy.is_shutdown():
    ret,frame=cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect the face
    rects = detector(gray, 1)
    # Detect landmarks for each face
    for rect in rects:
        # Get the landmark points
        shape = predictor(gray, rect)
        # Convert it to the NumPy Array
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape = shape_np

        # Display the landmarks
        for i, (x, y) in enumerate(shape):
            # Draw the circle to mark the keypoint 
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    # Display the image
    cv2.imshow('Landmark Detection', frame)
    msg3=bridge.cv2_to_imgmsg(frame,"bgr8")

    # Press the escape button to terminate the code
    pub.publish(msg3)
    rate.sleep()
    key=cv2.waitKey(30)
    if key==27:
        break
    if rospy.is_shutdown():
        cap.release()

