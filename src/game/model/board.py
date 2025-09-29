from typing import List, Mapping, Tuple
from game.constants import *
from game.model import Piece
from game.model import Move
from game.utils import within_bounds

import copy
import numpy as np


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
        self.king_side_castle = { WHITE : False, BLACK : False }
        self.queen_side_castle = { WHITE : False, BLACK : False }
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
            return False, lambda: 0
        
        # remove references
        self.pieces.remove(piece)
        self.team_and_type_map[piece.team][piece.piece_type].remove(piece)

        def undo_remove():
            self.coord_map[rank, file] = piece
            self.pieces.append(piece)
            self.team_and_type_map[piece.team][piece.piece_type].append(piece)

        return True, undo_remove
    

    def move_piece(self, move : Move):
        piece, to_coord, promote = move.piece, move.to_coord, move.promote

        # CACHE UPDATE
        self.cache.append(self.get_state(WHITE))
        def undo_cache():
            self.cache.pop()

        # EN PASSANT STATE UPDATE
        previous_en_passant = self.en_passant
        def undo_en_passant_state():
            self.en_passant = previous_en_passant

        if piece.piece_type == PAWN and piece.team == BLACK and piece.coord[0] == 1 and to_coord[0] == 3 or \
                piece.piece_type == PAWN and piece.team == WHITE and piece.coord[0] == 6 and to_coord[0] == 4:
            self.en_passant = to_coord
        else:
            self.en_passant = None

        # REMOVE AT TARGET LOCATION
        was_capture, undo_remove = self.remove_piece_at(*to_coord)
        if was_capture or piece.piece_type == PAWN:
            self.non_pawn_or_capture_moves = 0
        else:
            self.non_pawn_or_capture_moves += 1

        # ROOK MOVE IN CASE OF CASTLING
        king_rank = 0 if piece.team == BLACK else 7
        is_castle = piece.piece_type == KING and (piece.coord == [king_rank, 4]).all() and ((to_coord == [king_rank, 6]).all() or (to_coord == [king_rank, 2]).all())
        if is_castle:
            king_castle = (to_coord == [king_rank, 6]).all()
            assert (king_castle and self.king_side_castle[piece.team]) or (not king_castle and self.queen_side_castle[piece.team]), "castle move not available"
            old_rook_coord = np.array([king_rank, 7] if king_castle else [king_rank, 0])
            new_rook_coord = np.array([king_rank, 5] if king_castle else [king_rank, 3])
            rook_piece = self.coord_map.pop(tuple(old_rook_coord), None)
            assert rook_piece is not None and rook_piece.piece_type == ROOK and rook_piece.team == piece.team
            self.coord_map[*new_rook_coord] = rook_piece
            rook_piece.coord = new_rook_coord
        
        def undo_castle():
            if is_castle:
                self.coord_map[*old_rook_coord] = rook_piece
                self.coord_map.pop(tuple(new_rook_coord), None)
                rook_piece.coord = old_rook_coord

        # MOVE PIECE TO TARGET LOCATION
        old_piece_coord = piece.coord
        self.coord_map.pop(tuple(piece.coord), None)
        self.coord_map[*to_coord] = piece
        piece.coord = to_coord

        def undo_move():
            self.coord_map[*old_piece_coord] = piece
            self.coord_map.pop(tuple(to_coord), None)
            piece.coord = old_piece_coord

        pawn_dir = -1 if piece.team == WHITE else 1
        is_en_passant = previous_en_passant is not None and (previous_en_passant + [pawn_dir, 0] == to_coord).all() and piece.piece_type == PAWN
        if is_en_passant:
            _, undo_en_passant = self.remove_piece_at(*previous_en_passant)
        else:
            undo_en_passant = lambda: 0

        # PROMOTE IF PAWN IS MOVED TO FURTHEST RANK
        end_rank = 6 if piece.team == BLACK else 1
        assert (promote == -1) == (old_piece_coord[0] != end_rank) or piece.piece_type != PAWN, f"promotion error {promote}, {to_coord}, {piece}"
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
            for undo_func in [undo_en_passant_state, undo_cache, undo_move, undo_castle, undo_remove, undo_en_passant, undo_promote]:
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
        current_state = self.get_state(WHITE)
        for cached_state in self.cache:            
            # if the same position is found twice in cache, then the same position was reached 3 times in total, thus threefold repetition
            if (cached_state == current_state).all():
                count += 1
                if count >= 2:
                    return True

        return False
        

    def get_state(self, team : Team) -> np.array:
        # each channel (in dim 2) represents a team/piece_type pair
        state = np.zeros((16, 8, 8), dtype=np.float32)

        for piece in self.pieces:
            channel_index = (0 if piece.team == team else 8) + piece.piece_type.value
            state[channel_index, *piece.coord] = 1
        
        # States are observed such that the agent's and opponent's pieces are on the same side.
        if team == BLACK:
            state = state[:, ::-1, ::-1].copy()
        
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
        
        self.king_side_castle = { WHITE : True, BLACK : True }
        self.queen_side_castle = { WHITE : True, BLACK : True }