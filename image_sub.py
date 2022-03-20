#!/usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
import dlib
import numpy as np
import threading
bridge=CvBridge()
rospy.init_node('image2',anonymous=False)
def publisher_thread(msg1):
    rate = rospy.Rate(30) # ROS Rate at 5Hz
    while not rospy.is_shutdown():
        pub2.publish(msg1)
        rate.sleep()

def callback(msg):
    msg2=bridge.imgmsg_to_cv2(msg,"bgr8")
    cv2.imshow("im",msg2)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('/opt/pal/ferrum/shape_predictor_68_face_landmarks.dat')
    
    gray = cv2.cvtColor(msg2, cv2.COLOR_BGR2GRAY)
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
            cv2.circle(msg2, (x, y), 1, (0, 0, 255), -1)
    msg3=bridge.cv2_to_imgmsg(msg2,"bgr8")
    worker=threading.Thread(target=publisher_thread(msg3))
    worker.start()
pub2=rospy.Publisher('/webcam2',Image,queue_size=1)
sub=rospy.Subscriber('/webcam',Image,callback)
rospy.spin()

