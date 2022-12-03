# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

import numpy as np


def angle_clip(angle):
    # hard clip angle to [-pi, pi]
    return np.clip(angle, -np.pi, np.pi)

def angle_limit(angle):
    # soft limit yaw angle to [-pi, pi]
    if angle > np.pi:
        angle = 2 * np.pi - angle
    elif angle < -np.pi:
        angle = 2 * np.pi + angle
    return angle

def get_k_max(array, k):
    _k_sort = np.argpartition(array, -k)[-k:]  # 最大的k个数据的下标
    return array[_k_sort]

def get_k_min(array, k):
    _k_sort = np.argpartition(array, min(k, array.shape[0] - 1))[:k]  # 最小的k个数据的下标
    return array[_k_sort]

def get_arg_k_max(array, k):
    _k_sort = np.argpartition(array, -k)[-k:]  # 最大的k个数据的下标
    return _k_sort

def get_arg_k_min(array, k):
    _k_sort = np.argpartition(array, min(k, array.shape[0] - 1))[:k]  # 最小的k个数据的下标
    return _k_sort
