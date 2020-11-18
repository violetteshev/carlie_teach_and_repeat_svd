#!/usr/bin/env python


### IMPORT MODULES ###
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# In order to import OpenCV when using Python 3, need to remove ROS python2.7 dist packages.
if sys.version_info[0] == 3 and '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # so can import opencv for python3, silly ROS
import cv2 as cv

### ARGUMENTS AND MAIN ###
def ParseArguments():
    parser = argparse.ArgumentParser(description='Used to visualise a Carlie teach dataset')
    parser.add_argument('teach_dataset_file', metavar='T', type=str, nargs=1, help='path to a teach dataset text file (required)')
    parser.add_argument('--repeat_dataset_file', '-R', metavar='R', type=str, nargs=1, help='path to a teach dataset text file (optional)')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # Pass arguments
    args = ParseArguments()

    # Read in teach dataset file (CSV file) into numpy array
    # teach_dataset = [frame_id, relative_odom_x, relative_odom_y, relative_odom_yaw, relative_pose_x, relative_pose_y, relative_pose_yaw]
    teach_dataset = np.genfromtxt(args.teach_dataset_file[0], delimiter=', ', skip_header=1)
    teach_dataset[:,0] = np.arange(0, teach_dataset.shape[0]) # add in frame IDs to column 1, else will be NAN

    # Read in repeat dataset file (CSV file) into numpy array if provided
    if args.repeat_dataset_file != None:
        repeat_dataset = np.genfromtxt(args.repeat_dataset_file[0], delimiter=', ', skip_header=1)
        repeat_dataset[:,0] = np.arange(0, repeat_dataset.shape[0]) # add in frame IDs to column 1, else will be NAN


    # Plot
    fig = plt.figure(figsize=(8,8))
    plt.plot(teach_dataset[:,4], teach_dataset[:,5], color='b', marker='x', markersize=6, linestyle='-', linewidth=1, label='Teach Path')
    if args.repeat_dataset_file != None:
        plt.plot(repeat_dataset[:,4], repeat_dataset[:,5], color='r', marker='+', markersize=6, linestyle='-', linewidth=1, label='Repeat Path')

    plt.legend()
    plt.axis('equal')
    plt.xlabel('X-Position')
    plt.ylabel('Y-Position')
    plt.title('Carlie Teach and Repeat')
    plt.show()
