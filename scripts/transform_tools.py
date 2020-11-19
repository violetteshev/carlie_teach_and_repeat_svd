#!/usr/bin/env python

### IMPORT CLASSES ###
import numpy as np
import transforms3d as t3

### IMPORT MESSAGE TYPES ###
import geometry_msgs.msg as geometry_msgs

### TRANSFORM TOOLS ###
# because ROS TF sucks - take from https://bitbucket.org/acrv/ros_helpers/src/master/src/guiabot_tools.py
def _to_trans(position, orientation):
    return t3.affines.compose([position.x, position.y, position.z],
                              t3.quaternions.quat2mat([
                                  orientation.w, orientation.x, orientation.y,
                                  orientation.z
                              ]), [1, 1, 1])


def ang_diff(a, b):
    """Returns shortest angular distance from a to b"""
    return np.mod(b - a + np.pi, 2 * np.pi) - np.pi


def append_trans(a, b):
    """Returns the trans from appending b to a"""
    return np.matmul(a, b)


def dist_between_trans(a, b):
    """Returns xy distance between trans a and b"""
    return (np.sum((a[0:2, -1] - b[0:2, -1])**2))**0.5


def pose_msg_to_trans(msg):
    return _to_trans(msg.position, msg.orientation)


def diff_trans(a, b):
    """Returns the trans for going from a to b"""
    return np.matmul(np.linalg.inv(a), b)


def mean_trans(ts):
    """Returns the mean trans from a list of trans (rotation mean is only mean yaw)"""
    ys = [t3.euler.mat2euler(t[0:3, 0:3])[2] for t in ts]
    y_mean = np.arctan2(
        np.sum([np.sin(y) for y in ys]), np.sum([np.cos(y) for y in ys]))
    mean = np.mean(ts, axis=0)
    mean[0:3, 0:3] = t3.euler.euler2mat(0, 0, y_mean)
    return mean


def relative_yaw_to_trans(a, b):
    """Returns the relative yaw to b, from a"""
    return ang_diff(
        t3.euler.mat2euler(a[0:3, 0:3])[2],
        np.arctan2(b[1, -1] - a[1, -1], b[0, -1] - a[0, -1]))


def tf_msg_to_trans(msg):
    return _to_trans(msg.translation, msg.rotation)


def trans_to_pose_msg(trans):
    T, R, Z, S = t3.affines.decompose(trans)
    return geometry_msgs.Pose(
        position=geometry_msgs.Point(*T),
        orientation=geometry_msgs.Quaternion(
            *np.roll(t3.quaternions.mat2quat(R), -1)))


def trans_from_yaw(yaw, affine=None):
    yaw_rot = t3.euler.euler2mat(0, 0, yaw)
    if affine is None:
        return yaw_rot
    else:
        ret = np.copy(affine)
        ret[0:3, 0:3] = yaw_rot
        return ret


def trans_from_xyzrpy(x, y, z, roll, pitch, yaw):
    return t3.affines.compose([x, y, z], t3.euler.euler2mat(roll, pitch, yaw),
                              [1, 1, 1])


def yaw_from_trans(trans):
    T, R, Z, S = t3.affines.decompose(trans)
    return t3.euler.mat2euler(R)[2]


def yaw_from_pose_msg(msg):
    return yaw_from_trans(pose_msg_to_trans(msg))


def xyzrpy_from_trans(trans):
    T, R, Z, S = t3.affines.decompose(trans)
    e = t3.euler.mat2euler(R)
    return tuple(np.concatenate((T, e)))


def distance_of_trans(trans):
    """computes the 2D length of the transform"""
    res = np.sqrt(np.sum(np.power(trans[0:2,-1], 2)))
    return res