import numpy as np
from typing import List


class Move:
    to_coord : np.array

    def __init__(self, to_coord : np.array | List[int]):
        if type(to_coord) == np.array:
            self.to_coord = to_coord
        else:
            self.to_coord = np.array(to_coord)
    

    def __eq__(self, other : 'Move'):
        return (self.to_coord == other.to_coord).all()


    def __repr__(self):
        return str(self.to_coord)