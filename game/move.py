import numpy as np
from typing import List, Tuple
from constants import *
from model import Board
from king_state import get_king_state


def get_moves(board : Board, team : int) -> List[Tuple[int, np.array]]:
    """Returns list of (piece_index, coord), indicating which piece can be moved and where.
       Returns None if in checkmate, and [] if no moves are available (stalemate)."""

    moves = []

    # controlled squares around king, coords of pieces pinned to king, and the direction from which they are pinned
    controlled, pin_coords, pin_dirs = get_king_state(board, team)

    # squares populated by team
    team_pop = np.zeros((8, 8))
    for team_coord in board.coords[team]:
        team_pop[*team_coord] = 1

    # king moves
    king_rank, king_file = board.coords[team][0]
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
    

    return moves