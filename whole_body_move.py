#!/usr/bin/env python
import rospy
from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from std_msgs.msg import Duration
from geometry_msgs.msg import Twist
rospy.init_node('head_node',anonymous=True)
#publish head position using /head_controller_command topic
head_pub=rospy.Publisher('/head_controller/command',JointTrajectory,queue_size=1)
head_traj=JointTrajectory()
head_pt=JointTrajectoryPoint()
t=Duration()
torso_pub=rospy.Publisher('/torso_controller/command',JointTrajectory,queue_size=1)
torso_traj=JointTrajectory()
torso_pt=JointTrajectoryPoint()
head_cal=0
head_angle=0
head_pos=0
head_real_pos=0
torso_cal=0
torso_angle=0
torso_pos=0
torso_real_pos=0
print("Input the overall orientation")
inputs=raw_input()
total_angle=float(inputs)
print("the input is ",total_angle)
sign=0

while not rospy.is_shutdown():
    rate=rospy.Rate(1000)
    #head angle to joint controller calibrators
    head_cal=0.0138888888888889
    torso_cal=0.0138888888888889
    leg_cal=1
    if abs(total_angle)>0:
        sign=total_angle/abs(total_angle)
    if abs(total_angle)<=10:
        head_angle=total_angle
        torso_angle=0
        leg_angle=0
        head_vel=0.0005*sign
        torso_vel=0
    elif abs(total_angle)>10 and abs(total_angle)<45:
        head_angle=0.6*total_angle
        torso_angle=0.4*total_angle
        leg_angle=0
        head_vel=0.0005*sign
        torso_vel=0.0005*sign/2.25
    elif abs(total_angle)>45 and abs(total_angle)<=90:
        head_angle=0.5*total_angle
        torso_angle=0.3*total_angle
        leg_angle=0.2*total_angle
        head_vel=0.0005*sign
        torso_vel=0.0005*sign*4/9

    head_pos=head_angle*head_cal
    torso_pos=torso_angle*torso_cal

    #head inputs
    head_traj.joint_names=['head_1_joint','head_2_joint']
    head_pt.positions=[head_real_pos,0]
    t=5
    if abs(head_real_pos)<abs(head_pos):
        head_real_pos+=head_vel#0.0005
        print(head_real_pos)
    head_pt.time_from_start.nsecs=t
    head_traj.points=[head_pt,]
    head_pub.publish(head_traj)

    #torso inputs
    torso_traj.joint_names=['torso_1_joint','torso_2_joint']
    torso_pt.positions=[torso_real_pos,0]
    t=5
    if torso_real_pos<head_pos:
        torso_real_pos+=torso_vel#0.0005
    torso_pt.time_from_start.nsecs=t
    torso_traj.points=[torso_pt,]
    torso_pub.publish(torso_traj)
    rate.sleep()
rospy.spin()


