import numpy as np
from typing import List, Tuple
from constants import *
from model import Board, Move
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
            if within_bounds(to_rank, to_file) \
                    and controlled[controlled_rank, controlled_file] == 0 \
                    and team_pop[to_rank, to_file] == 0:
                moves.append(Move(0, np.array([to_rank, to_file])))

    # if king is in check
    if controlled[1, 1] == 1:
        return moves if len(moves) > 0 else None  # None if checkmate
    

    # pawn moves
    piece_index = board.type_locs[team, PAWN]
    for pawn_coord in board.piece_coords(team, PAWN):
        pin_dir = pin_map[*pawn_coord]
        rank_diff = 1 if team == BLACK else -1

        # forward moves
        if pin_map[*pawn_coord, 1] == 0:  # if not pinned in relevant direction
            if oppo_pop[*(pawn_coord + [rank_diff, 0])] == 0:
                moves.append(Move(piece_index, pawn_coord + [rank_diff, 0]))
            
                if (pawn_coord[0] == 1 and team == BLACK or pawn_coord[0] == 6 and team == WHITE) \
                        and oppo_pop[*(pawn_coord + [rank_diff * 2, 0])] == 0:
                    moves.append(Move(piece_index, pawn_coord + [rank_diff * 2, 0]))
                
        # capture moves
        moves.extend([Move(piece_index, pawn_coord + [rank_diff, file_diff]) 
                      for file_diff in [-1, 1]
                      if within_bounds(*(pawn_coord + [rank_diff, file_diff]))  # bounds checking
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
            moves.extend([Move(piece_index, to_coord) for to_coord in knight_coord + knight_diffs
                          if within_bounds(*to_coord)  # bounds checking
                          and team_pop[*to_coord] == 0])  # not populated by team
        
        piece_index += 1
    

    # bishop moves
    piece_index = board.type_locs[team, BISHOP]
    for bishop_coord in board.piece_coords(team, BISHOP):
        # determine directions in which bishop can move
        if (pin_map[*bishop_coord] == 0).all():
            bishop_dirs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        elif (pin_map[*bishop_coord] == [1, 1]).all() or (pin_map[*bishop_coord] == [-1, -1]).all():
            bishop_dirs = np.array([[-1, -1], [1, 1]])
        elif (pin_map[*bishop_coord] == [-1, 1]).all() or (pin_map[*bishop_coord] == [1, -1]).all():
            bishop_dirs = np.array([[1, -1], [-1, 1]])
        else:
            continue  # can't move if pinned from lateral direction

        moves.extend([Move(piece_index, to_coord) 
                      for to_coord in get_moves_along_directions(bishop_coord, bishop_dirs, team_pop, oppo_pop)])
    
        piece_index += 1
    

    # rook moves
    piece_index = board.type_locs[team, ROOK]
    for rook_coord in board.piece_coords(team, ROOK):
        # determine directions in which rook can move
        if (pin_map[*bishop_coord] == 0).all():
            rook_dirs = np.array([[-1, 0], [0, 1], [0, -1], [1, 0]])
        elif (pin_map[*bishop_coord] == [0, 1]).all() or (pin_map[*bishop_coord] == [0, -1]).all():
            rook_dirs = np.array([[0, -1], [0, 1]])
        elif (pin_map[*bishop_coord] == [-1, 0]).all() or (pin_map[*bishop_coord] == [1, 0]).all():
            rook_dirs = np.array([[1, 0], [-1, 0]])
        else:
            continue  # can't move if pinned from diagonal direction
    
        moves.extend([Move(piece_index, to_coord)
                      for to_coord in get_moves_along_directions(rook_coord, rook_dirs, team_pop, oppo_pop)])

        piece_index += 1
    

    # queen moves
    piece_index = board.type_locs[team, QUEEN]
    for queen_coord in board.piece_coords(team, QUEEN):
        if (pin_map[*bishop_coord] == 0).all():
            queen_dirs = np.array([[-1, 0], [0, 1], [0, -1], [1, 0], [-1, -1], [-1, 1], [1, -1], [1, 1]])
        else:
            queen_dirs = np.array([pin_map[*queen_coord], -pin_map[*queen_coord]])
        
        moves.extend([Move(piece_index, to_coord)
                      for to_coord in get_moves_along_directions(queen_coord, queen_dirs, team_pop, oppo_pop)])

        piece_index += 1


    return moves


def get_moves_along_directions(origin : np.array, dirs : np.array, team_pop : np.array, oppo_pop : np.array):
    """Gets all moves for piece that can move along given directions."""

    moves = []

    for dist in range(7):
        if len(dirs) == 0:
            break

        to_coords = origin + dirs * dist
        remove = []

        for dir_index, to_coord in enumerate(to_coords):
            if team_pop[*to_coord] or not within_bounds(*to_coord):
                remove.append(dir_index)
            elif oppo_pop[*to_coord]:
                moves.append(to_coord)
                remove.append(dir_index)
            else:
                moves.append(to_coord)
        
        dirs = np.delete(dirs, remove, axis=0)
    
    return moves