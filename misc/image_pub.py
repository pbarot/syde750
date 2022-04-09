#!/usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError

cap=cv2.VideoCapture(0)
print(cap.isOpened())
bridge=CvBridge()
pub=rospy.Publisher('/webcam',Image,queue_size=1)
rospy.init_node('image',anonymous=False)
rate=rospy.Rate(10)
while not rospy.is_shutdown():
    ret,frame=cap.read()
    msg=bridge.cv2_to_imgmsg(frame,"bgr8")
    pub.publish(msg)
    rate.sleep()
    key=cv2.waitKey(30)
    if key==27:
        break
    if rospy.is_shutdown():
        cap.release()

