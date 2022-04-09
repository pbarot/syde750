#!/usr/bin/env python2
import rospy
import numpy as np
from std_msgs.msg import String, Int32MultiArray, Int16MultiArray, Int16, Float32, Int32
#import localize_utils as lu
import warnings
import sys

import rospkg
import rospy
import moveit_commander
from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.msg import RobotTrajectory, RobotState
from geometry_msgs.msg import PoseStamped
import actionlib
from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryGoal, FollowJointTrajectoryAction, FollowJointTrajectoryResult, JointTolerance
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from aruco_msgs.msg import MarkerArray
from std_msgs.msg import String 
from reemc_joint_trajectory_player.joint_trajectory_client import JointTrajectoryClient


warnings.filterwarnings('ignore')

total_speaking_l = []
total_speaking_r = []

first = True

# right hand joints [0, 2.4, 2.5]

def generate_traj(total_speaking):
   # right arm joints: [1,2,3,4,5,6,7]
   # j = [1.48, 0, 1.4, 0.05, -1.6, 0, 0] for tup
   # j = [1.48, 0, -1.9, 0.05, -1.2, 0, 0] for tdown
   wait = 0
   avg = np.mean(np.array(total_speaking))
   print(avg)
   if(avg >= 0.1):
      # thumbs up position
      traj = [1.48, 0, 1.4, 0.05, -1.6, 0, 0]
      wait = 1
   elif(avg < 0.1):
      # thumb down position
      traj = [1.48, 0, -1.9, 0.05, -1.6, 0, 0]

   r_hand = [0.4, 2.4, 2.5]
   return traj, r_hand, wait

def listener():


   rospy.Subscriber("/speaker_tracking/speech_signal_total", Int32MultiArray, callback)

   print('IDENTIFYING SPEECH....')

   rospy.spin()

def actuate():
   global total_speaking_l
   global total_speaking_r

   global hand_r_jtc
   global hand_l_jtc

   global arm_r_jtc
   global arm_l_jtc

   global first


   if(len(total_speaking_l) == 10):
      traj_l, hand, wait_l= generate_traj(total_speaking_l)
      traj_r, hand, wait_r= generate_traj(total_speaking_r)
      time = 10


      if(first == True):
         arm_r_jtc.add_point(traj_r,time)
         arm_r_jtc.start()

         arm_l_jtc.add_point(traj_l,time)
         arm_l_jtc.start()

         hand_r_jtc.add_point(hand,time)
         hand_r_jtc.start()

         hand_l_jtc.add_point(hand,time)
         hand_l_jtc.start()
       
         first = False
         total_speaking_l= []
         total_speaking_r= []



      elif(first == False):
         arm_r_jtc.add_point(traj_r,0.5)
         arm_r_jtc.start()

         arm_l_jtc.add_point(traj_l,0.5)
         arm_l_jtc.start()


         if(wait_l == 1):
            arm_l_jtc.wait(0.4)

         if(wait_r == 1):
            arm_r_jtc.wait(0.4)

         arm_r_jtc.clear()
         arm_l_jtc.clear()

         total_speaking_l = []
         total_speaking_r = []


def callback(data):
   global hand_r_jtc
   global hand_l_jtc

   global arm_r_jtc
   global arm_l_jtc

   global first
   global total_speaking_l
   global total_speaking_r



   result = data.data
   total_speaking_l.append(result[0])
   total_speaking_r.append(result[1])

   actuate()
   

if __name__ == '__main__':

   rospy.init_node("reemc_thumbs")
   right_arm_controller = rospy.get_param("/right_arm_controller/joints")
   arm_r_jtc = JointTrajectoryClient('right_arm_controller', right_arm_controller)

   right_hand_controller = rospy.get_param("/right_hand_controller/joints")
   hand_r_jtc = JointTrajectoryClient('right_hand_controller', right_hand_controller)

   left_arm_controller = rospy.get_param("/left_arm_controller/joints")
   arm_l_jtc = JointTrajectoryClient('left_arm_controller', left_arm_controller)

   left_hand_controller = rospy.get_param("/left_hand_controller/joints")
   hand_l_jtc = JointTrajectoryClient('left_hand_controller', left_hand_controller)


   listener()

   rospy.on_shutdown(arm_l_jtc.stop)
   rospy.on_shutdown(arm_r_jtc.stop)

   rospy.on_shutdown(hand_l_jtc.stop)
   rospy.on_shutdown(hand_r_jtc.stop)





