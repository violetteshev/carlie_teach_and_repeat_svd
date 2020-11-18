#!/usr/bin/env python

### IMPORT CLASSES ###
import os
import rospy
import shutil
import cv2 as cv
import numpy as np
import tf_conversions
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion

### IMPORT MESSAGE TYPES ###
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry


### TEACH NODE CLASS ###
class RepeatNode():

    # INITIALISATION
    def __init__(self):
        # Parameters
        self.odom_topic_recieved = False
        self.frame_counter = 0
        self.frame_id = 0
        self.previous_odom = None # odometry pose of previous frame
        self.current_odom = None # odometry pose of current frame
        self.first_frame_odom = None # odometry of first frame

        # ROS Init Node
        rospy.init_node('repeat_node')
        rospy.loginfo("Repeat Node Initialised")

        # Constants
        self.PROCESS_EVERY_NTH_FRAME = rospy.get_param('~process_every_nth_frame', 1)
        self.TEACH_DATASET_FILE = rospy.get_param('~teach_dataset', '/home/nvidia/Documents/route_1_processed/dataset.txt')
        self.CV_BRIDGE = CvBridge()

        self.SAVE_REPEAT_DATA = rospy.get_param('~save_repeat_dataset', False)
        self.SAVE_IMAGE_RESIZE = (rospy.get_param('~save_image_resize_x', 640), rospy.get_param('~save_image_resize_y', 480))
        self.BASE_PATH = rospy.get_param('~base_path', '/home/nvidia/Documents')
        self.ROUTE_NAME = rospy.get_param('~route_name', 'route_1_processed')

        # Setup save directory and dataset file
        if self.SAVE_REPEAT_DATA:
            self.save_path = os.path.join(self.BASE_PATH, self.ROUTE_NAME)
            if os.path.exists(self.save_path):
                shutil.rmtree(self.save_path) # will delete existing save_path directory and its contents
            os.makedirs(self.save_path)
            self.dataset_file = open(os.path.join(self.save_path, 'dataset.txt'), 'w')
            self.dataset_file.write("Frame_ID, relative_odom_x(m), relative_odom_y(m), relative_odom_yaw(rad), relative_pose_x(m), relative_pose_y(m), relative_pose_yaw(rad)\n")

        # Get teach dataset
        # teach_dataset = [frame_id, relative_odom_x, relative_odom_y, relative_odom_yaw, relative_pose_x, relative_pose_y, relative_pose_yaw]
        self.teach_dataset = np.genfromtxt(self.TEACH_DATASET_FILE, delimiter=', ', skip_header=1)
        self.teach_dataset[:,0] = np.arange(0, self.teach_dataset.shape[0]) # add in frame IDs to column 1, else will be NAN

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
            rospy.logwarn('Waiting until odometry data is received. Make sure topic is published and topic name is correct.')
            return

        # Only process every nth frame
        self.frame_counter = (self.frame_counter + 1) % self.PROCESS_EVERY_NTH_FRAME
        if self.frame_counter != 0:
            return

        # Relative odometry from previous frame and relative pose from first frame
        if self.current_odom == None or self.previous_odom == None:
            rospy.logwarn('Unable to get relative odom transform. Make sure topic is published and topic name is correct.')
            relative_odom = None # to ensure all zeros

        elif self.frame_id != 0:
            current_odom_tf = tf_conversions.fromMsg(self.current_odom.pose.pose)

            # relative odometry
            previous_odom_tf = tf_conversions.fromMsg(self.previous_odom.pose.pose)
            relative_odom_tf = previous_odom_tf.Inverse() * current_odom_tf
            relative_odom = tf_conversions.toMsg(relative_odom_tf)

            # relative pose
            relative_pose_tf = self.first_frame_tf.Inverse() * current_odom_tf
            relative_pose = tf_conversions.toMsg(relative_pose_tf)

        else:
            # first frame 
            self.first_frame_odom = self.current_odom.pose.pose # set odom for first frame
            self.first_frame_tf = tf_conversions.fromMsg(self.first_frame_odom)
            relative_pose = Pose() # to ensure all zeros
            relative_odom = Pose() # to ensure all zeros
            

        # Attempt to convert ROS image into CV data type (i.e. numpy array)
        try:
            img_bgr = self.CV_BRIDGE.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr("Unable to convert ROS image into CV data. Error: " + str(e))
            return

        # Save repeat dataset if required
        if self.SAVE_REPEAT_DATA:
            # Resize image
            img_save = cv.resize(img_bgr, self.SAVE_IMAGE_RESIZE)

            # Save Image and Relative Odometry
            frame_name = "frame_%06d.png" % self.frame_id
            try:
                cv.imwrite(os.path.join(self.save_path, frame_name), img_save)
            except Exception as e:
                rospy.logwarn("Was unable to save image. Error: " + str(e))
                return

            # Write to dataset file
            if relative_odom != None:
                (odom_roll, odom_pitch, odom_yaw) = euler_from_quaternion([relative_odom.orientation.x, relative_odom.orientation.y, relative_odom.orientation.z, relative_odom.orientation.w])
                (pose_roll, pose_pitch, pose_yaw) = euler_from_quaternion([relative_pose.orientation.x, relative_pose.orientation.y, relative_pose.orientation.z, relative_pose.orientation.w])
                self.dataset_file.write('%s, %0.4f, %0.4f, %0.4f, %0.4f, %0.4f, %0.4f\n'%(frame_name, relative_odom.position.x, relative_odom.position.y, odom_yaw, relative_pose.position.x, relative_pose.position.y, pose_yaw))
            else:
                self.dataset_file.write('%s, Nan, Nan, Nan, Nan, Nan, Nan\n'%(frame_name))

        # Image Matching
        self.ImageMatching(img_bgr, relative_odom)

        # Update frame ID and previous odom
        self.frame_id += 1
        self.previous_odom = self.current_odom


    # IMAGE MATCHING
    def ImageMatching(self, img_bgr, relative_odom):
        pass


### MAIN ####
if __name__ == "__main__":
    try:
        repeat = RepeatNode()
    except rospy.ROSInterruptException:
        if repeat.SAVE_REPEAT_DATA:
            repeat.dataset_file.close()
    
    if repeat.SAVE_REPEAT_DATA:
        repeat.dataset_file.close()