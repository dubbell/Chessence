import numpy as np
from ..constants import *


class Piece:
    coord : np.array
    piece_type : PieceType
    team : Team

    def __init__(self, rank : int, file : int, piece_type : PieceType, team : Team):
        self.coord = np.array([rank, file], dtype=int)
        self.piece_type = piece_type
        self.team = team

    
    def __repr__(self):
        return f"{self.team} {self.piece_type} : {self.coord}"
