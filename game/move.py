import numpy as np
from typing import List, Tuple
from constants import *
from model import Board
from king_state import get_king_state
from utils import within_bounds


def get_moves(board : Board, team : int, en_passant : np.array = None) -> List[Tuple[int, np.array]]:
    """Returns list of (piece_index, coord), indicating which piece can be moved and where.
       Returns None if in checkmate, and [] if no moves are available (stalemate)."""

    moves = []

    # controlled squares around king, coords of pieces pinned to king, and the direction from which they are pinned
    controlled, pin_coords, pin_dirs = get_king_state(board, team)

    # lookup map of all pins, [0, 0] if no pin and direction if pin exists
    pin_map = np.zeros((8, 8, 2))
    for pin_coord, pin_dir in zip(pin_coords, pin_dirs):
        pin_map[*pin_coord] = pin_dir

    # squares populated by team
    team_pop = np.zeros((8, 8))
    for team_coord in board.coords[team]:
        team_pop[*team_coord] = 1

    # squares populated by opponent
    oppo_pop = np.zeros((8, 8))
    for oppo_coord in board.coords[int(not team)]:
        oppo_pop[*oppo_coord] = 1

    # king moves
    king_rank, king_file = board.coords[team][KING]
    for controlled_rank in range(3):
        for controlled_file in range(3):
            #king's square
            if controlled_rank == 1 and controlled_file == 1:
                continue
            to_rank, to_file = king_rank + controlled_rank - 1, king_file + controlled_file - 1
            if controlled[controlled_rank, controlled_file] == 0 \
                    and team_pop[to_rank, to_file] == 0:
                moves.append((0, np.array([to_rank, to_file])))

    # if king is in check
    if controlled[1, 1] == 1:
        return moves if len(moves) > 0 else None  # None if checkmate
    

    # pawn moves
    piece_index = board.type_locs[team, PAWN]
    for pawn_coord in board.piece_coords(team, PAWN):
        pin_dir = pin_map[*pawn_coord]
        rank_diff = 1 if team == BLACK else -1

        moves.extend([(piece_index, pawn_coord + [rank_diff, file_diff]) 
                      for file_diff in [-1, 1]
                      if within_bounds(pawn_coord + [rank_diff, file_diff])  # bounds checking
                      and ((pin_dir == 0).all()  # no pin
                           or (pin_dir == [rank_diff, file_diff]).all() or (-pin_dir == [rank_diff, file_diff]).all())  # or pin in same direction as move
                      and (oppo_pop[*(pawn_coord + [rank_diff, file_diff])] == 1  # populated by opponent
                           or (en_passant == pawn_coord + [0, file_diff]).all()  # or en passant
                           and (pin_map[*en_passant] == 0).all())])  # en passant'ed pawn can't be pinned

        piece_index += 1


    # knight moves
    piece_index = board.type_locs[team, KNIGHT]
    for knight_coord in board.piece_coords(team, KNIGHT):
        # if not pinned, then add moves
        if (pin_map[*knight_coord] == 0).all():
            moves.extend([(piece_index, to_coord) for to_coord in knight_coord + knight_diffs
                          if within_bounds(to_coord)  # bounds checking
                          and team_pop[*to_coord] == 0])  # not populated by team
        
        piece_index += 1
    



    

    return moves