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

    # number of consecutive moves that are made without captures or pawn moves
    non_pawn_or_capture_moves = 0
    
    # map from coordinates to pieces
    coord_map : Mapping[Tuple[int, int], Piece]
    
    # map from team and piece type to piece
    team_and_type_map : Mapping[Team, Mapping[PieceType, List[Piece]]]

    # castling rights
    king_side_castle : Mapping[Team, bool]
    queen_side_castle : Mapping[Team, bool]

    # whether previous move enables en passant, None or en passant coord
    en_passant : np.array

    def __init__(self):
        self.pieces = []
        self.cache = []
        self.king_side_castle = { WHITE : True, BLACK : True }
        self.queen_side_castle = { WHITE : True, BLACK : True }
        self.en_passant = None
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


    def remove_piece_at(self, rank : int, file : int) -> bool:
        """Removes piece at coord. True if there was a piece there, otherwise False. Also returns function to undo removal."""
        assert within_bounds(rank, file), "position out of bounds"
        piece = self.coord_map.pop((rank, file), None)

        if piece is None:
            return False, None
        
        # remove references
        self.pieces.remove(piece)
        self.team_and_type_map[piece.team][piece.piece_type].remove(piece)

        def undo_remove():
            self.coord_map[rank, file] = piece
            self.pieces.append(piece)
            self.team_and_type_map[piece.team][piece.piece_type].append(piece)

        return True, undo_remove
    

    def move_piece(self, piece : Piece, to_coord : np.array, promote : int = -1):
        # EN PASSANT STATE UPDATE
        previous_en_passant = self.en_passant.copy()
        def undo_en_passant():
            self.en_passant = previous_en_passant

        if piece.piece_type == PAWN and piece.team == BLACK and piece.coord[0] == 1 and to_coord[0] == 3 or \
                piece.piece_type == PAWN and piece.team == WHITE and piece.coord[0] == 6 and to_coord[0] == 4:
            self.en_passant = to_coord
        else:
            self.en_passant = None

        # CACHE UPDATE
        self.cache.append(copy.deepcopy(self.pieces))
        def undo_cache():
            self.cache.pop()

        # REMOVE AT TARGET LOCATION
        was_capture, undo_remove = self.remove_piece_at(*to_coord)
        if was_capture or piece.piece_type == PAWN:
            self.non_pawn_or_capture_moves = 0
        else:
            self.non_pawn_or_capture_moves += 1

        # MOVE PIECE TO TARGET LOCATION
        old_piece_coord = piece.coord.copy()
        self.coord_map.pop(piece.coord)
        self.coord_map[*to_coord] = piece
        piece.coord = to_coord

        def undo_move():
            self.coord_map[*piece.coord] = piece
            self.coord_map.pop(*to_coord)
            piece.coord = old_piece_coord

        # PROMOTE IF PAWN IS MOVED TO FURTHEST RANK
        promoted = False
        promotions = [QUEEN, ROOK, BISHOP, KNIGHT]
        if promote != -1 and piece.piece_type == PAWN:
            promotion = promotions[promote]
            self.team_and_type_map[piece.team][PAWN].remove(piece)
            self.team_and_type_map[piece.team][promotion].append(piece)
            piece.piece_type = promotion
            promoted = True
        
        def undo_promote():
            if promoted:
                piece.piece_type = PAWN
                self.team_and_type_map[piece.team][promotion].remove(piece)
                self.team_and_type_map[piece.team][PAWN].append(piece)

        def undo():
            for undo_func in [undo_en_passant, undo_cache, undo_remove, undo_move, undo_promote]:
                undo_func()

        return undo


    def check_50_move_rule(self):
        """Check for 50 move rule."""
        if self.non_pawn_or_capture_moves >= 50:
            return True
        return False


    def check_threefold(self):
        """Check for threefold repetition."""
        count = 0
        for cached_pieces in self.cache:
            if len(cached_pieces) != len(self.pieces):
                continue
            
            # if the same position is found twice in cache, then the same position was reached 3 times in total, thus threefold repetition
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
        state = np.zeros((16, 8, 8), dtype=np.float32)

        for piece in self.pieces:
            channel_index = (0 if piece.team == WHITE else 8) + piece.piece_type.value
            state[channel_index, *piece.coord] = 1
        
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