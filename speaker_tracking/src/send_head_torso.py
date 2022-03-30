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

#from gazebo_ros_link_attacher.srv import Attach, AttachRequest, AttachResponse

warnings.filterwarnings('ignore')

total_bearing = []
first = True

def generate_traj(total_bearing):
   # head joints: [yaw, pitch]
  
   wait = 0
   avg = np.mean(np.array(total_bearing))
   print(avg)

   traj = [math.radians(avg), 0]

   return traj

def listener():

   #rospy.init_node('listener', anonymous=True)

   rospy.Subscriber("/speaker_tracking/bearing", Int32, callback) 
   print('TRACKING....')

   rospy.spin()

def actuate():
   global total_bearing
   global head_jtc
   global torso_jtc
   global first

   v = math.radians(15)
   if(len(total_bearing) == 5):
      traj = generate_traj(total_bearing)

      h, t = traj[0]*0.3, traj[0]*0.7
      head_jtc.add_point([h,0], 0.5)
      head_jtc.start()   
      head_jtc.clear()

      torso_jtc.add_point([t,0], 0.5)
      torso_jtc.start()   
      torso_jtc.clear()
      total_bearing = []
      

def callback(data):
   global hand_jtc
   global arm_jtc
   global first
   global total_bearing

   #rospy.init_node("reemc_moveit_example")

   result = data.data
   total_bearing.append(result)
   actuate()
   

if __name__ == '__main__':

#   listener()    
   rospy.init_node("reemc_moveit")
   head_controller = rospy.get_param("/head_controller/joints")
   head_jtc = JointTrajectoryClient('head_controller', head_controller)

   torso_controller = rospy.get_param("/torso_controller/joints")
   torso_jtc = JointTrajectoryClient('torso_controller', torso_controller)
   print('got')
   listener()

   rospy.on_shutdown(head_jtc.stop)
   rospy.on_shutdown(torso_jtc.stop)
  #  HEAD_LOOK_DOWN = [0.0,0.0]

  # # result = data.data
  #  #total_angles.append(result)




