#!/usr/bin/env python
import rospy
from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from std_msgs.msg import Duration
rospy.init_node('head_node',anonymous=True)
def callback(msg):
    print("Sub called")
#publish head position using /head_controller_command topic
pub=rospy.Publisher('/head_controller/command',JointTrajectory,queue_size=1)
#subscribe head position using /head_controller/state
sub=rospy.Subscriber('/head_controller/state',JointTrajectoryControllerState,callback)
traj=JointTrajectory()
pt=JointTrajectoryPoint()
t=Duration()
pos=0
while not rospy.is_shutdown():
    rate=rospy.Rate(1000)
    traj.joint_names=['head_1_joint','head_2_joint']
    pt.positions=[pos,0]
    print(pos)
   #pt.velocities=[1000,0]
    #pt.accelerations=[0,0]
    t=5
    if pos<1.25:
        pos+=0.0005
    pt.time_from_start.nsecs=t
    traj.points=[pt,]
    pub.publish(traj)
    rate.sleep()
rospy.spin()

