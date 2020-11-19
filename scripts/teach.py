#!/usr/bin/env python

### IMPORT CLASSES ###
import os
import rospy
import shutil
import cv2 as cv
import transform_tools
from cv_bridge import CvBridge

### IMPORT MESSAGE TYPES ###
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry


### TEACH NODE CLASS ###
class TeachNode():

    # INITIALISATION
    def __init__(self):
        # Variables
        self.frame_id = 0
        self.previous_odom = None # odometry pose of previous frame
        self.current_odom = None # odometry pose of current frame
        self.first_frame_odom = None # odometry of first frame
        self.odom_topic_recieved = False

        # ROS Init Node
        rospy.init_node('teach_node')
        rospy.loginfo("Teach Node Initialised")

        # Constants
        self.KEYFRAME_DISTANCE_THRESHOLD = rospy.get_param('~keyframe_distance_threshold', 0.25)
        self.IMAGE_RESIZE = (rospy.get_param('~image_resize_x', 640), rospy.get_param('~image_resize_y', 480))
        self.BASE_PATH = rospy.get_param('~base_path', '/home/nvidia/Documents')
        self.ROUTE_NAME = rospy.get_param('~route_name', 'route_1')
        self.CV_BRIDGE = CvBridge()

        # Setup save directory and dataset file
        self.save_path = os.path.join(self.BASE_PATH, self.ROUTE_NAME)
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path) # will delete existing save_path directory and its contents
        os.makedirs(self.save_path)
        self.dataset_file = open(os.path.join(self.save_path, 'dataset.txt'), 'w')
        self.dataset_file.write("Frame_ID, relative_odom_x(m), relative_odom_y(m), relative_odom_yaw(rad), relative_pose_x(m), relative_pose_y(m), relative_pose_yaw(rad)\n")

        # ROS Subcribers
        self.odom_subscriber = rospy.Subscriber('odom', Odometry, self.Odom_Callback)
        self.image_subscriber = rospy.Subscriber('image_raw', Image, self.Image_Callback)

        # ROS Spin
        while not rospy.is_shutdown():
            pass

    # ODOM CALLBACK
    def Odom_Callback(self, data):
        self.odom_topic_recieved = True
        self.current_odom = data

        # set previous odom if has not already been set
        if self.previous_odom == None:
            self.previous_odom = data

    # IMAGE CALLBACK
    def Image_Callback(self, data):
        # Only start process images once odom callback has run once
        if not self.odom_topic_recieved:
            rospy.loginfo('Waiting until odometry data is received. Make sure topic is published and topic name is correct.')
            return

        if self.current_odom == None or self.previous_odom == None:
            rospy.logwarn('Unable to get relative pose transform. Make sure odometry topic is published and the teach node is subscribed to the correct topic.')
            return # safeguard

        # Relative odometry from previous frame and relative pose from first frame
        if self.frame_id != 0:
            current_odom_tf = transform_tools.pose_msg_to_trans(self.current_odom.pose.pose)

            # relative odometry
            previous_odom_tf = transform_tools.pose_msg_to_trans(self.previous_odom.pose.pose)
            relative_odom_tf = transform_tools.diff_trans(previous_odom_tf, current_odom_tf)
            relative_odom = transform_tools.trans_to_pose_msg(relative_odom_tf)

            # relative pose
            relative_pose_tf = transform_tools.diff_trans(self.first_frame_tf, current_odom_tf)
            relative_pose = transform_tools.trans_to_pose_msg(relative_pose_tf)

        else:
            # first frame 
            self.first_frame_odom = self.current_odom.pose.pose # set odom for first frame
            self.first_frame_tf = transform_tools.pose_msg_to_trans(self.first_frame_odom)

            relative_pose = Pose()
            relative_pose.orientation.w = 1
            relative_odom = relative_pose


        if self.frame_id >= 1 and transform_tools.distance_of_trans(relative_odom_tf) < self.KEYFRAME_DISTANCE_THRESHOLD:
            return
            
        # Attempt to convert ROS image into CV data type (i.e. numpy array)
        try:
            img_bgr = self.CV_BRIDGE.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr("Unable to convert ROS image into CV data. Error: " + str(e))
            return

        # Resize Image
        img_bgr = cv.resize(img_bgr, self.IMAGE_RESIZE)

        # Save Image and Relative Odometry
        frame_name = "frame_%06d.png" % self.frame_id
        try:
            cv.imwrite(os.path.join(self.save_path, frame_name), img_bgr)
        except Exception as e:
            rospy.logwarn("Was unable to save image. Error: " + str(e))
            return

        # Write to dataset file
        odom_yaw = transform_tools.yaw_from_pose_msg(relative_odom)
        pose_yaw = transform_tools.yaw_from_pose_msg(relative_pose)
        self.dataset_file.write('%s, %0.4f, %0.4f, %0.4f, %0.4f, %0.4f, %0.4f\n'%(frame_name, relative_odom.position.x, relative_odom.position.y, odom_yaw, relative_pose.position.x, relative_pose.position.y, pose_yaw))

        rospy.loginfo('Saved new teach frame: %d'%(self.frame_id))

        # Update frame ID and previous odom
        self.frame_id += 1
        self.previous_odom = self.current_odom


### MAIN ####
if __name__ == "__main__":
    try:
        teach = TeachNode()
    except rospy.ROSInterruptException:
        teach.dataset_file.close()

    teach.dataset_file.close()