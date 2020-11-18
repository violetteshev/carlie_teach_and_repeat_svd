#!/usr/bin/env python

### IMPORT CLASSES ###
import os
import math
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
from ackermann_msgs.msg import AckermannDriveStamped


### TEACH NODE CLASS ###
class RepeatNode():

    # INITIALISATION
    def __init__(self):
        # Variables
        self.update_visualisation = False
        self.current_matched_teach_frame_id = 0
        self.odom_topic_recieved = False
        self.frame_counter = 0
        self.frame_id = 0
        self.previous_odom = None # odometry pose of previous frame
        self.current_odom = None # odometry pose of current frame
        self.first_frame_odom = None # odometry of first frame
        # self.image_match_array = np.array((0,2)) # [repeat_frame_id, matched_teach_frame_id]

        # ROS Init Node
        rospy.init_node('repeat_node')
        rospy.loginfo("Repeat Node Initialised")

        # Constants
        self.FRAME_SEARCH_WINDOW = rospy.get_param('~frame_search_window', 3)
        self.PROCESS_EVERY_NTH_FRAME = rospy.get_param('~process_every_nth_frame', 1)
        self.TEACH_DATASET_FILE = rospy.get_param('~teach_dataset', '/home/nvidia/Documents/route_1_processed/dataset.txt')
        self.CV_BRIDGE = CvBridge()

        self.SAVE_REPEAT_DATA = rospy.get_param('~save_repeat_dataset', False)
        self.SAVE_IMAGE_RESIZE = (rospy.get_param('~save_image_resize_x', 640), rospy.get_param('~save_image_resize_y', 480))
        self.BASE_PATH = rospy.get_param('~base_path', '/home/nvidia/Documents')
        self.ROUTE_NAME = rospy.get_param('~route_name', 'route_1_processed')

        # Create OpenCV Windows
        cv.namedWindow('Repeat Image', cv.WINDOW_NORMAL)
        cv.namedWindow('Matched Teach Image', cv.WINDOW_NORMAL)

        # Setup save directory and dataset file
        if self.SAVE_REPEAT_DATA:
            self.save_path = os.path.join(self.BASE_PATH, self.ROUTE_NAME)
            if os.path.exists(self.save_path):
                shutil.rmtree(self.save_path) # will delete existing save_path directory and its contents
            os.makedirs(self.save_path)
            self.dataset_file = open(os.path.join(self.save_path, 'dataset.txt'), 'w')
            self.dataset_file.write("Frame_ID, relative_odom_x(m), relative_odom_y(m), relative_odom_yaw(rad), relative_pose_x(m), relative_pose_y(m), relative_pose_yaw(rad)\n")

        # Get teach dataset path and read dataset file
        # teach_dataset = [frame_id, relative_odom_x, relative_odom_y, relative_odom_yaw, relative_pose_x, relative_pose_y, relative_pose_yaw]
        self.teach_dataset_path = os.path.dirname(self.TEACH_DATASET_FILE)
        self.teach_dataset = np.genfromtxt(self.TEACH_DATASET_FILE, delimiter=', ', skip_header=1)
        self.teach_dataset[:,0] = np.arange(0, self.teach_dataset.shape[0]) # add in frame IDs to column 1, else will be NAN

        # ROS Subcribers
        self.odom_subscriber = rospy.Subscriber('odom', Odometry, self.Odom_Callback)
        self.image_subscriber = rospy.Subscriber('image_raw', Image, self.Image_Callback)

        # ROS Publishers
        self.ackermann_cmd_publisher = rospy.Publisher('/carlie/ackermann_cmd/autonomous', AckermannDriveStamped, queue_size=10)

		# Setup ROS Ackermann Drive Command Message
        self.ackermann_cmd = AckermannDriveStamped()
        self.ackermann_cmd.drive.steering_angle_velocity = 0.0 # see AckermannDriveStamped message for definition
        self.ackermann_cmd.drive.acceleration = rospy.get_param('acceleration', 0.5) # see AckermannDriveStamped message for definition
        self.ackermann_cmd.drive.jerk = 0 # see AckermannDriveStamped message for definition

        # ROS Spin
        while not rospy.is_shutdown():
            if self.update_visualisation:
                teach_img = cv.imread(os.path.join(self.teach_dataset_path, 'frame_%06d.png'%(self.current_matched_teach_frame_id)), cv.IMREAD_GRAYSCALE)
                cv.imshow('Repeat Image', self.img_proc)
                cv.imshow('Matched Teach Image', teach_img)
                cv.waitKey(1)
                self.update_visualisation = False

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
        match_teach_id, lateral_offset = self.ImageMatching(img_bgr, relative_odom)
        rospy.loginfo('Matched To Teach Frame ID: %d'%(match_teach_id))

        # Controller
        self.Controller(match_teach_id, lateral_offset)

        # Update frame ID and previous odom
        self.frame_id += 1
        self.previous_odom = self.current_odom


    # IMAGE MATCHING
    def ImageMatching(self, img_bgr, relative_odom):
        # Setup comparison temp variables
        best_score = None
        best_max_location = None

        # Preprocess repeat image
        self.img_proc = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
        self.img_proc = cv.resize(self.img_proc, (64, 48))

        # Take center patch of repeat image
        img_proc_patch = self.ImageCropCenter(self.img_proc, 0.6)

        # Loop through teach dataset within given search radius
        start_idx = int(max(self.current_matched_teach_frame_id-self.FRAME_SEARCH_WINDOW, 0))
        end_idx = int(min(self.current_matched_teach_frame_id+self.FRAME_SEARCH_WINDOW+1, self.teach_dataset.shape[0]))
        # rospy.loginfo('Start: %d, End: %d'%(start_idx, end_idx))
        for teach_frame_id in self.teach_dataset[start_idx:end_idx, 0]:

            # Read in teach processed img
            teach_img = cv.imread(os.path.join(self.teach_dataset_path, 'frame_%06d.png'%(teach_frame_id)), cv.IMREAD_GRAYSCALE)

            # Compare using normalised cross correlation (OpenCV Template Matching Function)
            result = cv.matchTemplate(teach_img, img_proc_patch, cv.TM_CCOEFF_NORMED)

            # Get maximum value and its location
            min_val, max_val, min_location, max_location = cv.minMaxLoc(result)

            if best_score == None or best_score < max_val:
                best_score = max_val
                best_max_location = max_location
                self.current_matched_teach_frame_id = int(teach_frame_id)

        # Max location is top_left want center
        patch_center_location = np.array([max_location[0], max_location[1]]) + np.array([img_proc_patch.shape[1]/2.0, img_proc_patch.shape[0]/2.0])
        lateral_offset = patch_center_location[0] - self.img_proc.shape[1] // 2

        # Update visualiation
        self.update_visualisation = True

        # return
        return self.current_matched_teach_frame_id, lateral_offset


    def ImageCropCenter(self, img, portion):
        # portion is from from 0 to 1 
        cropx = int(round(img.shape[1] * portion))
        cropy = int(round(img.shape[0] * portion))

        y,x = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)

        return img[starty:starty+cropy,startx:startx+cropx]

    # CONTROLLER
    def Controller(self, match_teach_id, lateral_offset):
        # CONSTANTS
        pose_frame_lookahead = 2
        lateral_offset_scale_factor = 0.1
        rho_gain = 0.6 # rho_gain > 0
        alpha_gain = -0.7 # (alpha_gain - rho_gain) > 0
        beta_gain = 0 # beta_gain < 0

        wheel_base = rospy.get_param('wheel_base', 0.312)
        max_foward_vel = rospy.get_param('max_forward_velocity', 0.5)
        max_steering_angle = rospy.get_param('max_steering_angle', math.radians(45.0))
        min_steering_angle = rospy.get_param('min_steering_angle', math.radians(-45.0))

        # Determine relative pose difference between current position and a teach frame specified look-ahead distance
        deltas = np.array([0, lateral_offset*lateral_offset_scale_factor, 0])
        deltas += self.teach_dataset[match_teach_id+1:match_teach_id+pose_frame_lookahead+1, 1:4].sum(axis=0)

        # Adapted From Peter Corke's Textbook - Driving a Car-Like Robot to a Pose (pg. 106)
        theta = 0 # impossible to analytically determine with single frame matching, assume minimal, so set to 0
        rho = np.sum(np.sqrt(np.power(deltas[0:2], 2)))
        alpha = np.arctan(deltas[1]/deltas[0]) - theta
        beta = -theta - alpha

        lin_vel = min(max(rho_gain * rho, 0), max_foward_vel)
        ang_vel = alpha_gain * alpha + beta_gain * beta
        
        if lin_vel != 0:
            steering_angle = np.arctan(ang_vel * wheel_base / lin_vel)
        else:
            steering_angle = 0
        steering_angle = min(max(steering_angle, min_steering_angle), max_steering_angle)

        # Set values and publish message
        self.ackermann_cmd.drive.speed = lin_vel
        self.ackermann_cmd.drive.steering_angle = steering_angle
        self.ackermann_cmd_publisher.publish(self.ackermann_cmd)


### MAIN ####
if __name__ == "__main__":
    try:
        repeat = RepeatNode()
    except rospy.ROSInterruptException:
        if repeat.SAVE_REPEAT_DATA:
            repeat.dataset_file.close()
    
    if repeat.SAVE_REPEAT_DATA:
        repeat.dataset_file.close()
