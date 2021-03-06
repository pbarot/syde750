import rospy
import cv2
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

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/stargazer/workspace/catkin_ws/shape_predictor_68_face_landmarks.dat')

#cap=cv2.VideoCapture("/home/stargazer/workspace/catkin_ws/speech_test_2.mp4")
cap=cv2.VideoCapture(6)

print(cap.isOpened())
# bridge=CvBridge()
pub=rospy.Publisher('/speaker_tracking/speech_signal',Int32,queue_size=1000)
rospy.init_node('voice_activity',anonymous=False)
rate=rospy.Rate(120)


def returnCameraIndexes():

    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr 

print(returnCameraIndexes())

homs =[]
woms = []
num_pix = []
a_r=[]
a_g=[]
a_b=[]


stats = np.zeros((7,1))

total_num_pix = []
total_height = []
total_lightness = []


curr_speak = []

count = 0
flag = 0

base_e = 0
base_h = 0
base_l = 0

curr_e = 0
curr_h = 0
curr_l = 0
while not rospy.is_shutdown():
#while(True):
    ret,frame=cap.read()
    
    if(ret == True):
        image = imutils.resize(frame, width = int(320), height = int(240))
        #image = cv2.flip(image, -1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        rects = detector(gray, 1)
        curr_stat = np.zeros((7,1))

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            
            shape = predictor(gray, rect)
            shape = fe.shape_to_np(shape)

            mouth_mask = fe.mouth(shape, image)
            # get features
            height = fe.mouth_height(shape, image)/fe.full_height(shape, image)
            lf = fe.lower_face(shape, image)

            # i = np.zeros((image.shape[0], image.shape[1]))
            # i[lf] = 1

            pixels = np.sum(mouth_mask)/np.sum(lf)
            lightness = np.mean(hsv[:,:,1][mouth_mask])/np.mean(hsv[:,:,1][lf])

            total_num_pix.append(pixels)
            total_height.append(height)
            total_lightness.append(int(lightness))

            if(len(total_num_pix) == 5 and flag ==0):
                base_e = fe.process_npix(total_num_pix, base_e, flag)
                base_h = fe.process_heights(total_height, base_h, flag)
                base_l = fe.process_lightness(total_lightness, base_l, flag)

                energy = 0
                flag = 1
                total_num_pix = []
                total_height = []
                total_lightness = []


            elif(len(total_num_pix) == 15 and flag == 1):
                curr_e, energy_p = fe.process_npix(total_num_pix,base_e,  flag)
                curr_h, energy_h = fe.process_heights(total_height,base_h,  flag)
                curr_l, energy_l = fe.process_lightness(total_lightness, base_l, flag)


                #curr_speak.append(np.bitwise_and(np.bitwise_and(curr_e, curr_h), curr_l))
                #print(np.bitwise_and(np.bitwise_and(curr_e, curr_h), curr_l))

                total_num_pix = []
                total_height = []
                total_lightness = []
                print(curr_h, curr_l)

        if(flag == 1):
            msg = Int32()
            if(curr_e !=None):
                msg.data = np.bitwise_and(curr_h, curr_l)# np.bitwise_and(np.bitwise_and(curr_h, curr_l), curr_l)
                pub.publish(msg)
        #out.write(image)
        # Display the image
        
        # cv2.putText(img = image, text = 'SPEAK: {}'.format(curr_e),org = (50,50), fontScale = 2, color = (0,0,255), fontFace = 1)
        cv2.imshow('mouth mask',image)

        key=cv2.waitKey(30)
        if key==27:
            break
        if rospy.is_shutdown():
            cap.release()
    else:
        break
    
    count = count +1
    #print(count)

cap.release()
#out.release()
cv2.destroyAllWindows()


#np.save('data.npy', stats)