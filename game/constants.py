import numpy as np


HALF_DIRS = np.array([[0, -1], [-1, -1], [-1, 0], [-1, 1]])

ALL_DIRS = np.vstack((HALF_DIRS, -HALF_DIRS))