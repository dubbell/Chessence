import numpy as np
from typing import List

from utils import out_of_bounds, get_starting_board
from search import cast_rays, is_controlled_by, locate, locate_first, knight_jumps

WHITE, BLACK = np.arange(2)
TEAM_SLICES = [slice(0, 6), slice(6, None)]
PAWN, KING, QUEEN, ROOK, KNIGHT, BISHOP = np.arange(6)
TEAM, TYPE, RANK, FILE = np.arange(4)

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

FROM = slice(0, 2)
TO = slice(2, None)


class Move:
    def __init__(self, from_pos : np.array, to_pos : np.array, piece_type : int):
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.piece_type = piece_type


# move calculation -------

def get_pawn_moves(pawn_pos, team_board_pop : np.array, opp_board_pop : np.array, team : int) -> List[np.array]:
    pawn_moves = []
    
    move_dir = 1 if team == BLACK else -1

    candidate_poss = []

    pawn_rank, pawn_file = pawn_pos

    new_pos = new_rank, new_file = np.array([pawn_rank + move_dir, pawn_file - 1])
    if not out_of_bounds(new_pos) and opp_board_pop[new_rank, new_file]:
        candidate_poss.append(new_pos)
    
    new_pos = new_rank, new_file = np.array([pawn_rank + move_dir, pawn_file + 1])
    if not out_of_bounds(new_pos) and opp_board_pop[new_rank, new_file]:
        candidate_poss.append(new_pos)
    
    new_pos = new_rank, new_file = np.array([pawn_rank + move_dir, pawn_file])
    if not opp_board_pop[new_rank, new_file] and not team_board_pop[new_rank, new_file]:
        candidate_poss.append(new_pos)
        new_pos = new_rank, new_file = np.array([pawn_rank + move_dir * 2, pawn_file])
        if ((team == WHITE and pawn_rank == 6 or team == BLACK and pawn_rank == 1) 
            and not opp_board_pop[new_rank, new_file] and not team_board_pop[new_rank, new_file]):
            candidate_poss.append(new_pos)

    candidate_poss = np.array(candidate_poss)
    if len(candidate_poss) == 0:
        return []
    candidate_diffs = candidate_poss - pawn_pos

    # now checking for pins

    # one_dirs is directions from which a pin can exist, lateral and diagonal
    one_dirs = np.array([[0, -1], [-1, -1], [-1, 0], [-1, 1]])
    # opposite directions of the ones in one_dirs
    two_dirs = -one_dirs

    # relevant piece types for corresponding cast
    relevant_types = [[ROOK, QUEEN] if np.abs(dir).sum() == 1 else [BISHOP, QUEEN] 
                      for dir in np.vstack((one_dirs, two_dirs))]

    # results of searching in one_dirs and two_dirs
    one_casts = cast_rays(board, pawn_pos, one_dirs)
    two_casts = cast_rays(board, pawn_pos, two_dirs)
    
    for candidate_pos, candidate_diff in zip(candidate_poss, candidate_diffs):
        if np.abs(candidate_diff).sum() == 2:
            candidate_diff = candidate_diff / 2
        # casts for which a pin would prevent movement in candidate_dir
        pin_casts = np.logical_not(
            np.logical_or(
                (candidate_diff == one_dirs).all(axis=1), 
                (candidate_diff == two_dirs).all(axis=1)))

        if pawn_file == 5:
            print(candidate_diff, pin_casts)
    
        # true if there's no pin, add pawn move
        if not np.any([list(forward[:2]) == [team, KING] and list(backward[:2]) in [[int(not team), t] for t in types]
                    or list(backward[:2]) == [team, KING] and list(forward[:2]) in [[int(not team), t] for t in types]
                       for forward, backward, types in zip(one_casts[pin_casts], two_casts[pin_casts], relevant_types)
                       if forward is not None and backward is not None]):
        
            pawn_moves.append(candidate_pos)
    
    return pawn_moves

def get_king_moves(board : np.array, king_pos : np.array, team_board_pop : np.array, team : int) -> List[np.array]:
    king_moves = []

    for pos_diff in [np.array([rank, file]) 
                     for rank in [0, -1, 1] 
                     for file in [0, -1, 1]
                     if file != 0 and rank != 0]:
        new_pos = new_rank, new_file = king_pos + pos_diff
        
        if out_of_bounds(new_pos) or team_board_pop[new_rank, new_file] or is_controlled_by(board, new_pos, not team):
            continue
        else:
            king_moves.append(new_pos)
    
    return king_moves

def get_line_moves(board : np.array, 
                   dirs : np.array,
                   piece_pos : np.array,
                   team_board_pop : np.array, 
                   opp_board_pop : np.array, 
                   team : int) -> List[Move]:
    line_moves = []

    # one_dirs is one half of the lateral and diagonal directions
    one_dirs = np.array([[0, -1], [-1, -1], [-1, 0], [-1, 1]])
    # opposite directions of the ones in one_dirs
    two_dirs = -one_dirs

    # relevant piece types for corresponding cast (pin check)
    relevant_types = [[ROOK, QUEEN] if np.abs(dir).sum() == 1 else [BISHOP, QUEEN] 
                      for dir in np.vstack((one_dirs, two_dirs))]

    # ray casts to check for pins
    one_casts = cast_rays(board, piece_pos, one_dirs)  # check forwards
    two_casts = cast_rays(board, piece_pos, two_dirs)  # check backwards

    # check directions (applies both ways for each) in which the piece is allowed to move, based on pins
    remaining_dirs = np.array([True for _ in range(len(dirs))])
    for dir_i, candidate_dir in enumerate(dirs):
        # casts for which a pin would prevent movement in candidate_dir
        pin_casts = np.logical_not(
            np.logical_or(
                (candidate_dir == one_dirs).all(axis=1),
                (candidate_dir == two_dirs).all(axis=1)))
        
        # if there's a pin in any of the pin_casts, then disallow movement in candidate_dir
        if np.any([list(forward[:2]) == [team, KING] and list(backward[:2]) in [[int(not team), t] for t in types]
                    or list(backward[:2]) == [team, KING] and list(forward[:2]) in [[int(not team), t] for t in types]
                for forward, backward, types in zip(one_casts[pin_casts], two_casts[pin_casts], relevant_types)
                if forward is not None and backward is not None]):
        
            remaining_dirs[dir_i] = False   
            
    # search and add moves along allowed directions until edge of board or a piece is found
    steps = 1
    while remaining_dirs.any():
        poss = piece_pos + steps * dirs[remaining_dirs]
        steps += 1
        for pos_i, pos in zip(np.arange(len(dirs))[remaining_dirs], poss):
            pos_rank, pos_file = pos
            if out_of_bounds(pos) or team_board_pop[pos_rank, pos_file]:
                remaining_dirs[pos_i] = False
                continue
            elif opp_board_pop[pos_rank, pos_file]:
                remaining_dirs[pos_i] = False
            
            line_moves.append(pos)
            
    
    return np.array(line_moves)

def get_moves(board : np.array, team : int):

    team_board = board[TEAM_SLICES[team]]
    opp_board = board[TEAM_SLICES[int(not team)]]

    # boolean matrix, whether position is populated
    team_board_pop = team_board.sum(axis=0) > 0
    opp_board_pop = opp_board.sum(axis=0) > 0

    available_moves = []  # to be computed

    # pawn moves
    for pawn_pos in locate(team_board[PAWN]):
        available_moves.extend([Move(pawn_pos, to_pos, PAWN) for to_pos in get_pawn_moves(pawn_pos, team_board_pop, opp_board_pop, team)])

    # king moves
    king_pos = locate_first(team_board[KING])
    available_moves.extend([Move(king_pos, to_pos, KING) for to_pos in get_king_moves(board, king_pos, team_board_pop, team)])

    # queen, rook, bishop moves
    for piece_type, dirs in ([QUEEN, ROOK, BISHOP], [np.vstack((LATERAL_DIRS, DIAGONAL_DIRS)), LATERAL_DIRS, DIAGONAL_DIRS]):
        for piece_pos in locate(team_board[piece_type]):
            available_moves.extend([Move(piece_pos, to_pos, piece_type) for to_pos in get_line_moves(board, dirs, piece_pos, team_board_pop, opp_board_pop, team)])

    return available_moves
            



board, white_board, black_board = get_starting_board()
white_board[PAWN][6, 3] = 0
white_board[PAWN][6, 2] = 0
white_board[QUEEN][5, 6] = 1
# black_board[BISHOP][5, 3] = 1

white_board_pop = white_board.sum(axis = 0) > 0
black_board_pop = black_board.sum(axis = 0) > 0

test = np.zeros((8, 8))

# queen_dirs = np.vstack((DIAGONAL_DIRS, LATERAL_DIRS))
# queen_poss = locate(white_board[QUEEN])
# for queen_pos in queen_poss:
#     rank, file = queen_pos
#     test[rank, file] = 1
#     for move in get_line_moves(board, queen_dirs, queen_pos, white_board_pop, black_board_pop, WHITE):
#         from_rank, from_file, to_rank, to_file = move
#         test[to_rank, to_file] = 2

# pawn_poss = locate(white_board[PAWN])
# for pawn_pos in pawn_poss:
#     rank, file = pawn_pos
#     test[rank, file] = 1
#     for move in get_pawn_moves(pawn_pos, white_board_pop, black_board_pop, WHITE):
#         _, _, to_rank, to_file = move
#         test[to_rank, to_file] = 2

# for i in range(8):
#     for j in range(8):
#         if is_controlled_by(board, np.array([i, j]), BLACK):
#             test[i, j] = 1


for i, j in knight_jumps(np.array([1, 4])):
    print(i, j)
    test[i, j] = 1


print(test)