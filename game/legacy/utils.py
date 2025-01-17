import numpy as np
from typing import Union, Tuple

PAWN, KING, QUEEN, ROOK, KNIGHT, BISHOP = np.arange(6)



def within_bounds(*args : Union[Tuple[int], Tuple[np.array]]) -> bool:
    """True if position is within the bounds of the chess board."""
    rank, file = args[0] if len(args) == 1 else args
    return rank >= 0 and rank <= 7 and file >= 0 and file <= 7

def out_of_bounds(*args : Union[Tuple[int], Tuple[np.array]]) -> bool:
    """Complement of within_bounds."""
    return not within_bounds(args)

def get_starting_board() -> np.array:
    """Returns board with pieces on starting positions."""

    board = np.zeros((12, 8, 8))
    white_board = board[:6, :, :]
    black_board = board[6:, :, :]

    white_board[PAWN, 6, :] = 1
    white_board[KING, 7, 4] = 1
    white_board[QUEEN, 7, 3] = 1
    white_board[ROOK, 7, [0, 7]] = 1
    white_board[KNIGHT, 7, [1, 6]] = 1
    white_board[BISHOP, 7, [2, 5]] = 1
    
    black_board[PAWN, 1, :] = 1
    black_board[KING, 0, 4] = 1
    black_board[QUEEN, 0, 3] = 1
    black_board[ROOK, 0, [0, 7]] = 1
    black_board[KNIGHT, 0, [1, 6]] = 1
    black_board[BISHOP, 0, [2, 5]] = 1

    return board, white_board, black_board