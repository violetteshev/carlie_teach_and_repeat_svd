#!/usr/bin/env python

### IMPORT CLASSES ###
import os
import math
import rospy
import shutil
import cv2 as cv
import numpy as np
import transform_tools
from cv_bridge import CvBridge
from tf.transformations import quaternion_from_euler


### IMPORT MESSAGE TYPES ###
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


### MAIN ####
if __name__ == "__main__":

    quat_45deg = quaternion_from_euler(0,0,math.pi/4)
    
    # Should return identity transform
    pose = Pose()
    pose.orientation.w = 1
    trans_eye = transform_tools.pose_msg_to_trans(pose)
    print('Identity Transform')
    print(transform_tools.xyzrpy_from_trans(trans_eye))
    print(trans_eye)

    # Should give me a 1m translation in x-axis
    pose = Pose()
    pose.position.x = 1
    pose.orientation.w = 1
    trans_1x = transform_tools.pose_msg_to_trans(pose)
    print('\n1m X-Axis Translation')
    print(transform_tools.xyzrpy_from_trans(trans_1x))
    print(trans_1x)

    # Should give me a 1m translation in y-axis
    pose = Pose()
    pose.position.y = 1
    pose.orientation.w = 1
    trans_1y = transform_tools.pose_msg_to_trans(pose)
    print('\n1m Y-Axis Translation')
    print(transform_tools.xyzrpy_from_trans(trans_1y))
    print(trans_1y)

    # Going from trans_1y to trans_1x should give me 1m in x and y direction
    trans_1y_to_1x = transform_tools.diff_trans(trans_1y, trans_1x)
    print('\nFrom trans_1y to trans_1x')
    print(transform_tools.xyzrpy_from_trans(trans_1y_to_1x))
    print(trans_1y_to_1x)


    # Should give me a 1m translation in x-axis and 45 degrees
    pose = Pose()
    pose.position.x = 1
    pose.orientation.x = quat_45deg[0]
    pose.orientation.y = quat_45deg[1]
    pose.orientation.z = quat_45deg[2]
    pose.orientation.w = quat_45deg[3]
    trans_1x45deg = transform_tools.pose_msg_to_trans(pose)
    print('\n1m X-Axis 45 Deg Translation')
    print(transform_tools.xyzrpy_from_trans(trans_1x45deg))
    print(trans_1x45deg)

    # Going from trans_1y to trans_1x45deg should give me 1m in x and y direction
    trans_1y_to_1x45deg = transform_tools.diff_trans(trans_1y, trans_1x45deg)
    print('\nFrom trans_1y to trans_1x45deg')
    print(transform_tools.xyzrpy_from_trans(trans_1y_to_1x45deg))
    print(trans_1y_to_1x45deg)

