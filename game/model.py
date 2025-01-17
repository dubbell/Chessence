from enum import Enum


class Team(Enum):
    WHITE = 0
    BLACK = 1

class Type(Enum):
    KING = 0
    QUEEN = 1
    ROOK = 2
    BISHOP = 3
    KNIGHT = 4
    PAWN = 5

class Piece:
    team : Team
    type : Type

    def __init__(self, team : Team = None, type : Type = None):
        self.team = team
        self.type = type

    def __repr__(self):
        if self.team is None or self.type is None:
            return "  "
        return (("W" if self.team == Team.WHITE else "B") 
              + ("N" if self.type == Type.KNIGHT else self.type.name[0]))
    
    def __eq__(self, other):
        return other.team == self.team and other.type == self.type

print(Piece(Team.BLACK, Type.ROOK) == Piece(Team.BLACK, Type.ROOK))
