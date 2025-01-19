import numpy as np
from typing import List
from constants import *


class Board:
    coords : List[np.array] # list of 2 arrays, each (X, 2) of corresponding piece coords
    types : List[np.array]  # list of 2 arrays, each of length X representing corresponding piece type
    type_locs = np.array # (2, 6) array, each element representing the starting index of piece in types








def get_moves(board : Board):

    # start with king moves
    pass