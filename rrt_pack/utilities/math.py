# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

import numpy as np


def angle_clip(angle):
    return np.clip(angle, -np.pi, np.pi)

def get_k_max(array, k):
    _k_sort = np.argpartition(array, -k)[-k:]  # 最大的k个数据的下标
    return array[_k_sort]


def get_k_min(array, k):
    _k_sort = np.argpartition(array, k)[:k]  # 最小的k个数据的下标
    return array[_k_sort]
