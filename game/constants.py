import numpy as np


HALF_DIRS = np.array([[0, -1], [-1, -1], [-1, 0], [-1, 1]])

ALL_DIRS = np.vstack((HALF_DIRS, -HALF_DIRS))

DIAG_DIRS = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])

LAT_DIRS = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])