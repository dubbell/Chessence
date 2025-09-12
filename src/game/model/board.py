from typing import List, Mapping, Tuple, Self
from ..constants import *
from .piece import Piece
from .move import Move
from ..utils import within_bounds
import copy


PIECE_TYPE_ORDER = [ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK]


class Board:
    # all pieces on the board
    pieces : List[Piece]

    # cache of previous positions, to check for threefold
    cache : List[List[Piece]]
    
    # map from coordinates to pieces
    coord_map : Mapping[Tuple[int], Piece]
    
    # map from team and piece type to piece
    team_and_type_map : Mapping[Team, Mapping[PieceType, List[Piece]]]


    def __init__(self):
        self.pieces = []
        self.cache = []
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


    def get_king(self, team : Team) -> Piece:
        king_pieces = self.of_team_and_type(team, KING)
        if len(king_pieces) > 0:
            return king_pieces[0]
        else:
            return None


    def of_team(self, team : Team) -> List[Piece]:
        return [piece
                for piece_list in self.team_and_type_map[team].values()
                for piece in piece_list]


    def of_team_and_type(self, team : Team, piece_type : PieceType) -> List[Piece]:
        return self.team_and_type_map[team][piece_type]


    def add_piece(self, piece_type : PieceType, team : Team, rank : int, file : int) -> Piece:
        assert within_bounds(rank, file), "position out of bounds"
        assert self.coord_map.get((rank, file)) is None, f"Piece add error: coordinate {(rank, file)} already occupied."

        piece = Piece(rank, file, piece_type, team)

        # add references
        self.pieces.append(piece)
        self.team_and_type_map[team][piece_type].append(piece)
        self.coord_map[rank, file] = piece

        return piece


    def remove_piece_at(self, rank : int, file : int):
        assert within_bounds(rank, file), "position out of bounds"
        piece = self.coord_map.pop((rank, file), None)

        if piece is None:
            return

        # remove references
        self.pieces.remove(piece)
        self.team_and_type_map[piece.team][piece.piece_type].remove(piece)
    

    def move_piece(self, piece : Piece, move : Move):
        if piece.piece_type == PAWN and piece.team == BLACK and piece.coord[0] == 1 and move.to_coord[0] == 3 or \
                piece.piece_type == PAWN and piece.team == WHITE and piece.coord[0] == 6 and move.to_coord[0] == 4:
            en_passant = move.to_coord
        else:
            en_passant = None

        self.cache.append(copy.deepcopy(self.pieces))

        self.remove_piece_at(*move.to_coord)
        self.coord_map.pop(tuple(piece.coord))
        self.coord_map[*move.to_coord] = piece
        piece.coord = move.to_coord

        return en_passant


    def check_threefold(self):
        """Check for threefold repetition."""
        count = 0
        for cached_pieces in self.cache:
            if len(cached_pieces) != len(self.pieces):
                continue
            
            if np.all([cached_piece == current_piece for cached_piece, current_piece in zip(cached_pieces, self.pieces)]):
                count += 1
                if count >= 2:
                    return True

        return False


    def move_to_new_board(self, piece : Piece, move : Move) -> Self:
        copied_board = copy.deepcopy(self)
        copied_piece = copied_board.coord_map[*piece.coord]
        copied_board.move_piece(copied_piece, move)
        return copied_board
        

    def get_state(self) -> np.array:
        # each channel (in dim 2) represents a team/piece_type pair
        state = np.zeros((1, 16, 8, 8), dtype=np.float32)

        for piece in self.pieces:
            channel_index = (0 if piece.team == WHITE else 8) + piece.piece_type.value
            state[0, channel_index, *piece.coord] = 1
        
        return state
    

    def reset(self):
        """Reset board to initial positions."""
        while len(self.pieces) > 0:
            self.remove_piece_at(*self.pieces[0].coord)

        self.cache = []

        for file in range(8):
            self.add_piece(PIECE_TYPE_ORDER[file], BLACK, 0, file)
            self.add_piece(PAWN, BLACK, 1, file)
            self.add_piece(PIECE_TYPE_ORDER[file], WHITE, 7, file)
            self.add_piece(PAWN, WHITE, 6, file)