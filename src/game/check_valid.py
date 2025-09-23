from game.model import Board
from game.constants import *
from game.utils import within_bounds

import numpy as np



__all__ = ["can_castle", "is_valid"]

king_side_diffs = [[0, 2]]
queen_side_diffs = [[0, -2], [0, -3]]

def is_controlled(board : Board, coord : np.array, opponent : Team):
    """Check whether `coord` is controlled by `opponent` pieces."""
    # KING CHECK
    opponent_king = board.get_king(opponent)
    if opponent_king is not None and (np.abs(coord - board.get_king(opponent).coord) <= 1).all():
        return True
    
    # PAWN CHECK
    pawns = board.of_team_and_type(opponent, PAWN)
    pawn_direction = -1 if opponent == WHITE else 1
    if np.any([(coord == pawn.coord + [[pawn_direction, -1], [pawn_direction, 1]]).all(axis=1) for pawn in pawns]):
        return True
    
    # KNIGHT CHECK
    if np.any([tuple(maybe_knight_coord) in board.coord_map
               and board.coord_map[*maybe_knight_coord].piece_type == KNIGHT 
               and board.coord_map[*maybe_knight_coord].team == opponent
               for maybe_knight_coord in coord + knight_diffs 
               if within_bounds(*maybe_knight_coord)]):
        return True
    
    # DIRECTION CHECK: QUEENS, ROOKS, BISHOPS
    for dir in directions:
        is_diagonal = not np.equal(dir, 0).any() # diagonal vs lateral movement
        cur_coord = coord + dir
        while within_bounds(*cur_coord):
            piece = board.coord_map.get(tuple(cur_coord))
            if piece is None:
                cur_coord += dir
                continue
            elif piece.team == opponent and piece.piece_type in [QUEEN, BISHOP if is_diagonal else ROOK]:
                return True
            else:
                break
    
    return False


def is_valid(board : Board, team : Team):
    """Check if current position is valid after a move made by `team`. Determined by whether `team` king is in check."""
    king = board.get_king(team)
    if king is None:
        return True
    opponent = other_team(team)
    return not is_controlled(board, king.coord, opponent)


def can_castle(board : Board, team : Team):
    """Check king/queen castling rights for `team`."""
    king_rank = 0 if team == BLACK else 7
    opponent = other_team(team)
    king_castle, queen_castle = board.king_side_castle[team], board.queen_side_castle[team]

    if king_castle:
        for coord in np.array([[king_rank, file] for file in range(4, 7)]):
            if board.coord_map.get(*coord) is not None or is_controlled(board, coord, opponent):
                king_castle = False
                break

    if queen_castle:
        for coord in np.array([[king_rank, file] for file in range(2, 5)]):
            if board.coord_map.get(*coord) is not None or is_controlled(board, coord, opponent):
                king_castle = False
                break
    
    return king_castle, queen_castle