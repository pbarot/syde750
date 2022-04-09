#!/usr/bin/env python2
import rospy
import numpy as np
from std_msgs.msg import String, Int32MultiArray, Int16MultiArray, Int16, Float32, Int32
#import localize_utils as lu
import warnings
import sys
import math


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

total_bearing = []
first = True

def rad(angle):
   return math.radians(angle)

def map_angles(angle):
   global prev
   diff = abs(angle) - abs(prev)

   if (diff <=15):
     head_angle=rad(angle)
     torso_angle=0
   
   elif (diff >15 and diff <45):
     head_angle=0.7*rad(angle)
     torso_angle=0.3*rad(angle)

   elif (diff >= 45):
     head_angle=0.6*rad(angle)
     torso_angle=0.4*rad(angle)


   return head_angle, torso_angle

def generate_traj(total_bearing):
   # head joints: [yaw, pitch]
   global prev

   wait = 0
   total_angle = np.array(total_bearing)[-1]
   print(total_angle)

   traj = [rad(total_angle), 0]
   if((abs(total_angle) - abs(prev) > 10)):
      head_angle, torso_angle = map_angles(total_angle)
   else:
      head_angle, torso_angle = map_angles(prev)

   head_traj = [head_angle, 0]
   torso_traj = [torso_angle, 0]

   return head_traj, torso_traj

def listener():

   rospy.Subscriber("/speaker_tracking/yaw_angle", Int32, callback) 
   print('TRACKING....')

   rospy.spin()

def actuate():
   global total_bearing
   global head_jtc
   global torso_jtc
   global first

   v = math.radians(15)
   if(len(total_bearing) == 5):
      head_traj, torso_traj = generate_traj(total_bearing)

      head_jtc.add_point(head_traj, 0.5)
      head_jtc.start()   
      head_jtc.clear()

      torso_jtc.add_point(torso_traj, 0.5)
      torso_jtc.start()   
      torso_jtc.clear()
      total_bearing = []
      

def callback(data):
   global hand_jtc
   global arm_jtc
   global first
   global total_bearing


   result = data.data
   total_bearing.append(result)
   actuate()
   

if __name__ == '__main__':
   prev = 0

   rospy.init_node("reemc_tracking")
   head_controller = rospy.get_param("/head_controller/joints")
   head_jtc = JointTrajectoryClient('head_controller', head_controller)

   torso_controller = rospy.get_param("/torso_controller/joints")
   torso_jtc = JointTrajectoryClient('torso_controller', torso_controller)

   listener()

   rospy.on_shutdown(head_jtc.stop)
   rospy.on_shutdown(torso_jtc.stop)
  



