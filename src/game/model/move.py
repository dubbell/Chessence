import numpy as np
from typing import List
from game.model import Piece


class Move:
    piece : Piece
    to_coord : np.array
    promote : int  # -1, 0, 1, 2, 3

    def __init__(self, piece : Piece, to_coord : np.array | List[int], promote : int = -1):
        if isinstance(to_coord, np.ndarray):
            self.to_coord = to_coord
        else:
            self.to_coord = np.array(to_coord)
        self.piece = piece
        self.promote = promote

    def __eq__(self, other : 'Move'):
        return (self.to_coord == other.to_coord).all() and self.piece == other.piece and self.promote == other.promote

    def __repr__(self):
        return str(self.to_coord)