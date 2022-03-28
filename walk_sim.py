#!/usr/bin/env python
import rospy
from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import Twist
from std_msgs.msg import Duration
rospy.init_node('twist_muxx',anonymous=True)
#def callback(msg):
    #print("Sub called")
#publish head position using /head_controller_command topic
pub=rospy.Publisher('/walking_controller/cmd_vel',Twist,queue_size=1)
pos=0
twist=Twist()
while not rospy.is_shutdown():
    rate=rospy.Rate(1000)
    twist.linear.x=3
    pub.publish(twist)
    rate.sleep()
rospy.spin()


