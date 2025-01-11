import numpy as np
from collections.abc import Callable
from typing import List

WHITE = 0
BLACK = 1

TEAM_SLICES = [slice(0, 6), slice(6, None)]

PAWN = 0
KING = 1
QUEEN = 2
ROOK = 3
KNIGHT = 4
BISHOP = 5

TEAM = 0
TYPE = 1
RANK = 2
FILE = 3




class Move:
    def __init__(self, from_pos : np.array, to_pos : np.array, piece_type : int):
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.piece_type = piece_type


# searching ----------
    
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

def resolve_piece(square : np.array):
    """Given square, returns TEAM, PIECE_TYPE."""
    index = square.argmax()
    return index // 6, index % 6


def cast_rays(board : np.array, pos : np.array, dirs : np.array, search_condition : Callable[[np.array], bool] = None):
    """Search from given position by casting rays in the directions given by dirs. search_condition is function
        that specifies what is being searched for. If not None, then cast_ray returns True when an instance is found.
        If search_condition is None, then cast_ray returns all pieces that are encountered when searching dirs.
        
        Returns True/False if search_condition is not None
                List of [team, piece_type, rank, file] if search_condition is None"""
    
    found_pieces = [None for _ in range(len(dirs))]
    remaining_dirs = [True for _ in range(len(dirs))]

    for distance in range(1, 8):
        poss = pos + distance * dirs[remaining_dirs]

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
                    if search_condition(board[:, pos_rank, pos_file]):
                        return True
                # if there is no search_condition, then collect the piece and position
                else:
                    found_pieces[dir_i] = [team, piece_type, pos_rank, pos_file]
                
                remaining_dirs[dir_i] = False
    
    return np.array(found_pieces, dtype = object) if search_condition is None else False




def is_controlled_by(board : np.array, pos : np.array, team : int):
    pass


# bounds checking -----------

def within_bounds(pos : np.array):
    """True if position is within the bounds of the chess board."""
    return (pos >= 0).all() and (pos <= 7).all()

def out_of_bounds(pos : np.array):
    """Complement of within_bounds."""
    return not within_bounds(pos)
    

# other utils -------

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


# move calculation -------

def get_pawn_moves(team_board : np.array, team_board_pop : np.array, opp_board_pop : np.array, team : int):
    pawn_moves = []

    pawn_poss = locate(team_board[PAWN])
    for pawn_pos in pawn_poss:
        move_dir = 1 if team == BLACK else -1

        # candidate move directions
        candidate_diffs = np.array([[move_dir, -1], [move_dir, 0], [move_dir, 1]])
        # candidate destination positions
        candidate_poss = pawn_pos + candidate_diffs

        to_delete = []

        # disregard first diagonal direction if no opponent at position
        diag_rank1, diag_file1 = candidate_poss[0]
        if not opp_board_pop[diag_rank1, diag_file1]:
            to_delete.append(0)

        # disregard moving forward if position already populated
        lat_rank, lat_file = candidate_poss[1]
        if opp_board_pop[lat_rank, lat_file] or team_board_pop[lat_rank, lat_file]:
            to_delete.append(1)
        
        # disregard second diagonal direction if no opponent at position
        diag_rank2, diag_file2 = candidate_poss[2]
        if not opp_board_pop[diag_rank2, diag_file2]:
            to_delete.append(2)
        
        # Delete disregarded positions and directions. If empty, move on to next pawn.
        candidate_diffs = np.delete(candidate_diffs, to_delete, axis=0)
        candidate_poss = np.delete(candidate_poss, to_delete, axis=0)
        if len(candidate_poss) == 0:
            continue

        # now checking for pins

        # one_dirs is directions from which a pin can exist, lateral and diagonal
        one_dirs = np.array([[0, -1], [-1, -1], [-1, 0], [-1, 1]])
        # opposite directions of the ones in one_dirs
        two_dirs = -one_dirs

        # relevant piece types for corresponding cast
        relevant_types = [[ROOK, QUEEN] if i % 2 == 0 else [BISHOP, QUEEN] for i in range(4)]

        # results of searching in one_dirs and two_dirs
        one_casts = cast_rays(pawn_pos, one_dirs)
        two_casts = cast_rays(pawn_pos, two_dirs)
        
        for candidate_pos, candidate_diff in zip(candidate_poss, candidate_diffs):
            relevant_casts = np.logical_not(
                np.logical_or(
                    (candidate_diff == one_dirs).all(axis=1), 
                    (candidate_diff == two_dirs).all(axis=1)))
        
            # true if there's no pin in any relevant direction, in which case the candidate move is added to the results list
            if not np.any([list(forward[:2]) == [team, KING] and list(backward[:2]) in [[int(not team), t] for t in types]
                                or list(backward[:2]) == [team, KING] and list(forward[:2]) in [[int(not team), t] for t in types]
                           for forward, backward, types in zip(one_casts[relevant_casts], two_casts[relevant_casts], relevant_types)]):
            
                pawn_moves.append(Move(pawn_pos, candidate_pos, PAWN))
        
        return pawn_moves

def get_king_moves(board : np.array, team_board : np.array, team_board_pop : np.array, team : int):
    king_moves = []

    king_pos = locate_first(team_board[KING])
    for pos_diff in [np.array([rank, file]) for rank in [0, -1, 1] for file in [0, -1, 1]]:
        new_pos = new_rank, new_file = king_pos + pos_diff
        
        if out_of_bounds(new_pos) or team_board_pop[new_rank, new_file] or is_controlled_by(board, new_pos, not team):
            continue
        else:
            king_moves.append(Move(king_pos, new_pos, KING))
    
    return king_moves

def get_queen_moves(board : np.array, 
                    team_board : np.array, 
                    team_board_pop : np.array, 
                    opp_board_pop : np.array, 
                    team : int) -> List[Move]:
    queen_moves = []

    queen_poss = locate(team_board[QUEEN])
    for queen_pos in queen_poss:
        # one_dirs is one half of the lateral and diagonal directions
        one_dirs = np.array([[0, -1], [-1, -1], [-1, 0], [-1, 1]])
        # opposite directions of the ones in one_dirs
        two_dirs = -one_dirs

        # relevant piece types for corresponding cast
        relevant_types = [[ROOK, QUEEN] if i % 2 == 0 else [BISHOP, QUEEN] for i in range(4)]

        # results of searching in one_dirs and two_dirs
        one_casts = cast_rays(board, queen_pos, one_dirs)
        two_casts = cast_rays(board, queen_pos, two_dirs)

        # check directions (applies both ways for each) in which the queen is allowed to move, based on pins
        allowed_dirs = np.array([True for _ in range(4)])
        for dir_i, candidate_dir in enumerate(one_dirs):
            relevant_casts = np.logical_not(
                np.logical_or(
                    (candidate_dir == one_dirs).all(axis=1),
                    (candidate_dir == two_dirs).all(axis=1)))
            
            # true if there's no pin in any relevant direction, in which case the candidate move is added to the results list
            if np.any([list(forward[:2]) == [team, KING] and list(backward[:2]) in [[int(not team), t] for t in types]
                        or list(backward[:2]) == [team, KING] and list(forward[:2]) in [[int(not team), t] for t in types]
                    for forward, backward, types in zip(one_casts[relevant_casts], two_casts[relevant_casts], relevant_types)
                    if forward is not None and backward is not None]):
            
                allowed_dirs[dir_i] = False        
                
        # search and add moves along allowed directions until edge of board or a piece is found
        steps = 1
        remaining_dirs = np.array([allowed_dirs[i % 4] for i in range(8)])
        while remaining_dirs.any():
            poss = queen_pos + steps * np.vstack((one_dirs, two_dirs))[remaining_dirs]
            steps += 1
            for pos_i, pos in zip(np.arange(8)[remaining_dirs], poss):
                pos_rank, pos_file = pos
                if out_of_bounds(pos) or team_board_pop[pos_rank, pos_file]:
                    remaining_dirs[pos_i] = False
                    continue
                elif opp_board_pop[pos_rank, pos_file]:
                    remaining_dirs[pos_i] = False
                
                queen_moves.append(Move(queen_pos, pos, QUEEN))
            
    
    return np.array(queen_moves)



def get_moves(board : np.array, team : int):

    team_board = board[TEAM_SLICES[team]]
    opp_board = board[TEAM_SLICES[int(not team)]]

    team_board_pop = team_board.sum(axis=0) > 0
    opp_board_pop = opp_board.sum(axis=0) > 0

    available_moves = []  # to be computed

    available_moves.extend(get_pawn_moves(team_board, team_board_pop, opp_board_pop, team))
    available_moves.extend(get_king_moves(board, team_board, team_board_pop, team))
    available_moves.extend(get_queen_moves(board, team_board, team_board_pop, opp_board_pop, team))
    

    return available_moves
            



board, white_board, black_board = get_starting_board()
white_board[PAWN][6, 3] = 0
white_board[PAWN][6, 2] = 0
white_board[QUEEN][5, 6] = 1

white_board_pop = white_board.sum(axis = 0) > 0
black_board_pop = black_board.sum(axis = 0) > 0

test = np.zeros((8, 8))
test[7, 3] = 1
test[5, 6] = 1

for move in get_queen_moves(board, white_board, white_board_pop, black_board_pop, WHITE):
    test[move.to_pos[0], move.to_pos[1]] = 2

print(test)