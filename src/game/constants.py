from enum import Enum


class Team(Enum):
    WHITE = 1
    BLACK = -1

    def __eq__(self, other):
        return isinstance(other, Enum) and self.value == other.value
    
    def __hash__(self):
        return self.value
    


WHITE = Team.WHITE
BLACK = Team.BLACK

def other_team(team : Team):
    return WHITE if team == BLACK else BLACK


class PieceType(Enum):
    KING = 0
    QUEEN = 1
    ROOK = 2
    BISHOP = 3
    KNIGHT = 4
    PAWN = 5

    def __eq__(self, other):
        return isinstance(other, Enum) and self.value == other.value
    
    def __hash__(self):
        return self.value
    
    

KING = PieceType.KING
QUEEN = PieceType.QUEEN
ROOK = PieceType.ROOK
BISHOP = PieceType.BISHOP
KNIGHT = PieceType.KNIGHT
PAWN = PieceType.PAWN

knight_diffs = [[rank_diff, file_diff] 
                for rank_diff in [-2, -1, 1, 2]
                for file_diff in [-2, -1, 1, 2]
                if abs(rank_diff) != abs(file_diff)]

directions = [[rank_diff, file_diff]
              for rank_diff in range(-1, 2)
              for file_diff in range(-1, 2)
              if rank_diff != 0 or file_diff != 0]