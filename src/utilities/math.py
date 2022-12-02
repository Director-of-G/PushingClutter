# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

import numpy as np


def angle_clip(angle):
    return np.clip(angle, -np.pi, np.pi)
