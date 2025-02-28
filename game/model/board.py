from constants import *
from typing import List, Mapping, Tuple
from piece import Piece


class Board:
    # all pieces on the board
    pieces : List[Piece]

    # map from coordinates to pieces
    coord_map : Mapping[Tuple[int], Piece]
    
    # map from team and piece type to piece
    team_and_type_map : Mapping[Team, Mapping[PieceType, List[Piece]]]
    

    def __init__(self):
        self.pieces = []
        self.coord_map = {}
        self.team_and_type_map = \
            { team : { piece_type : [] 
                    for piece_type in [KING, QUEEN, ROOK, BISHOP, KNIGHT, PAWN] } 
            for team in [WHITE, BLACK] }
        

    def __repr__(self):
        board = np.array([["  " for _ in range(8)] for _ in range(8)])
        for piece in self.pieces:
            name = ("W" if piece.team == WHITE else "B") + \
                    ("K" if piece.piece_type == KING else 
                    "Q" if piece.piece_type == QUEEN else
                    "R" if piece.piece_type == ROOK else
                    "B" if piece.piece_type == BISHOP else 
                    "N" if piece.piece_type == KNIGHT else "P")
            
            board[*piece.coord] = name

        return "[" + "]\n[".join([" ".join(rank) for rank in board]) + "]"


    def get_king(self, team : Team):
        king_pieces = self.of_team_and_type(team, KING)
        if len(king_pieces) > 0:
            return king_pieces[0]
        else:
            return None


    def of_team(self, team : Team):
        return [piece
                for piece_list in self.team_and_type_map[team].values()
                for piece in piece_list]


    def of_team_and_type(self, team : Team, piece_type : PieceType):
        return self.team_and_type_map[team][piece_type]


    def add_piece(self, piece_type : PieceType, team : Team, rank : int, file : int):
        assert self.coord_map.get((rank, file)) is None, f"Piece add error: coordinate {(rank, file)} already occupied."

        piece = Piece(rank, file, piece_type, team)

        # add references
        self.pieces.append(piece)
        self.team_and_type_map[team][piece_type].append(piece)
        self.coord_map[rank, file] = piece

        return piece


    def remove_piece_at(self, rank : int, file : int):
        piece = self.coord_map.pop((rank, file), None)

        assert piece is not None, f"Piece remove error: coordinate {(rank, file)} already empty."

        # remove references
        self.pieces.remove(piece)
        self.team_and_type_map[piece.team][piece.piece_type].remove(piece)
        self.coord_map