from utils import within_bounds, out_of_bounds
import numpy as np
from collections.abc import Callable

WHITE, BLACK = np.arange(2)
PAWN, KING, QUEEN, ROOK, KNIGHT, BISHOP = np.arange(6)
TEAM_SLICES = [slice(0, 6), slice(6, None)]

LATERAL_DIRS = np.array(
    [(rank_diff, file_diff) 
    for rank_diff in [-1, 0, 1]
    for file_diff in [-1, 0, 1]
    if abs(rank_diff) != abs(file_diff)])

DIAGONAL_DIRS = np.array(
    [(rank_diff, file_diff) 
    for rank_diff in [-1, 0, 1]
    for file_diff in [-1, 0, 1]
    if abs(rank_diff) + abs(file_diff) == 2])


def locate(sub_board : np.array) -> np.array:
    """Returns list of coordinates for each piece found on the (sub-)board."""
    board_sum = sub_board if len(sub_board.shape) == 2 else sub_board.sum(axis=0)
    flat_indices = np.arange(64)[board_sum.flatten() > 0]
    return np.array(np.unravel_index(flat_indices, (8, 8))).T

def locate_first(piece_states : np.array) -> np.array:
    """Faster version of locate. Only takes a single piece's state and looks for first instance. Used for kings."""
    for rank in range(8):
        for file in range(8):
            if piece_states[rank, file] > 0:
                return np.array([rank, file])

def knight_jumps(pos : np.array):
    """How the knight can jump from a given position"""
    return np.array([knight_pos for knight_pos in pos + 
                     np.array([(rank_diff, file_diff) 
                               for rank_diff in [-2, -1, 1, 2]
                               for file_diff in [-2, -1, 1, 2]
                               if abs(rank_diff) != abs(file_diff)])
                     if within_bounds(knight_pos)])

def resolve_piece(square : np.array):
    """Given square, returns TEAM, PIECE_TYPE."""
    index = square.argmax()
    return index // 6, index % 6

def cast_rays(board : np.array, cast_pos : np.array, dirs : np.array, search_condition : Callable[[int, int], bool] = None):
    """Search from given position by casting rays in the directions given by dirs. search_condition is function
        that specifies what is being searched for. If not None, then cast_ray returns True when an instance is found.
        If search_condition is None, then cast_ray returns all pieces that are encountered when searching dirs.
        
        Returns True/False if search_condition is not None
                List of [team, piece_type, rank, file] if search_condition is None"""
    
    found_pieces = [None for _ in range(len(dirs))]
    remaining_dirs = [True for _ in range(len(dirs))]

    for distance in range(1, 8):
        poss = cast_pos + distance * dirs[remaining_dirs]

        for dir_i, pos in zip(np.arange(len(dirs))[remaining_dirs], poss):
            pos_rank, pos_file = pos

            # check if edge of board is reached
            if out_of_bounds(pos):
                remaining_dirs[dir_i] = False
                continue
            # if piece is found
            if board[:, pos_rank, pos_file].any():
                # check piece type and team
                team, piece_type = resolve_piece(board[:, pos_rank, pos_file])
                # if something is being searched for, then check the search_condition and return True if it is met
                if search_condition is not None:
                    if search_condition(team, piece_type):
                        return True
                # if there is no search_condition, then collect the piece and position
                else:
                    found_pieces[dir_i] = [team, piece_type, pos_rank, pos_file]
                
                remaining_dirs[dir_i] = False
    
    return np.array(found_pieces, dtype = object) if search_condition is None else False


def is_controlled_by(board : np.array, pos : np.array, team : int):
    """Checks whether position is controlled by given team."""
    rank, file = pos
    team_board = board[TEAM_SLICES[team]]


    if ((rank <= 6 and team == WHITE or rank >= 1 and team == BLACK) and 
            np.any(team_board[PAWN][
                [rank + 1 if team == WHITE else -1 for _ in range(2)], 
                [threat for threat in [file - 1, file + 1] if threat >= 0 and threat <= 7]])):
        return True
        
    king_ranks, king_files = np.vstack((LATERAL_DIRS, DIAGONAL_DIRS)).T
    if np.any(team_board[KING][king_ranks, king_files]):
        return True
    
    return (cast_rays(board, pos, LATERAL_DIRS, lambda found_team, found_type: found_team == team and found_type in [ROOK, QUEEN]) or 
            cast_rays(board, pos, DIAGONAL_DIRS, lambda found_team, found_type: found_team == team and found_type in [BISHOP, QUEEN]))