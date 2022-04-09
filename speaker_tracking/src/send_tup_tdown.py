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

#from gazebo_ros_link_attacher.srv import Attach, AttachRequest, AttachResponse

warnings.filterwarnings('ignore')

total_speaking = []
first = True

# right hand joints [0, 2.4, 2.5]

def generate_traj(total_speaking):
   # right arm joints: [1,2,3,4,5,6,7]
   # j = [1.48, 0, 1.4, 0.05, -1.6, 0, 0] for tup
   # j = [1.48, 0, -1.9, 0.05, -1.2, 0, 0] for tdown
   wait = 0
   avg = np.mean(np.array(total_speaking))
   if(avg > 0.4 and avg < 0.8):
      print('YES!')
      # thumbs up position
      traj = [1.48, 0, 1.4, 0.05, -1.6, 0, 0]
      wait = 1
   elif(avg <=0.4 or avg >=0.8):
      print('NO!')
      # thumb down position
      traj = [1.48, 0, -1.9, 0.05, -1.6, 0, 0]

   r_hand = [0.4, 2.4, 2.5]
   return traj, r_hand, wait

def listener():

   #rospy.init_node('listener', anonymous=True)

   rospy.Subscriber("/speaker_tracking/speech_signal", Int32, callback) 
   print('IDENTIFYING SPEECH....')

   rospy.spin()

def actuate():
   global total_speaking
   global hand_jtc
   global arm_jtc
   global first


   if(len(total_speaking) == 10):
      traj, hand, wait= generate_traj(total_speaking)
         
      if(first == True):
         arm_jtc.add_point(traj,0.5)
         arm_jtc.start()
        # arm_jtc.wait(1.0)
         #arm_jtc.clear()

         hand_jtc.add_point(hand,0.5 )
         hand_jtc.start()
         #hand_jtc.wait(0.1)
        # hand_jtc.clear()

         #rospy.sleep(0.1) #sleep for a few seconds to stabilize
       #  rospy.loginfo("1")
         first = False
         total_speaking = []

         print('set...')
      elif(first == False):
         arm_jtc.add_point(traj,0.5)
         arm_jtc.start()
         if(wait == 1):
            arm_jtc.wait(0.4)
         arm_jtc.clear()

         # if(wait == 1):
         #    rospy.sleep(0.4) #sleep for a few seconds to stabilize
       #  print('done ...')
         total_speaking = []

def callback(data):
   global hand_jtc
   global arm_jtc
   global first
   global total_speaking

   #rospy.init_node("reemc_moveit_example")

   result = data.data
   total_speaking.append(result)
   actuate()
   

if __name__ == '__main__':

#   listener()    
   rospy.init_node("reemc_moveit_example")
   right_arm_controller = rospy.get_param("/right_arm_controller/joints")
   arm_jtc = JointTrajectoryClient('right_arm_controller', right_arm_controller)

   right_hand_controller = rospy.get_param("/right_hand_controller/joints")
   hand_jtc = JointTrajectoryClient('right_hand_controller', right_hand_controller)

   listener()

   rospy.on_shutdown(arm_jtc.stop)
   rospy.on_shutdown(hand_jtc.stop)
  #  HEAD_LOOK_DOWN = [0.0,0.0]

  # # result = data.data
  #  #total_angles.append(result)




