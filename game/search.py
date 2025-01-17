import numpy as np
from model import Team, Type, Piece
from typing import List


def locate_first(board : np.array, team : Team, type : Type):
    for rank in range(8):
        for file in range(8):
            if board[rank, file] == Piece(team, type):
                return np.array([rank, file])

def locate(board : np.array, team : Team, types : List[Type]):
    np.array([(rank, file)
              for rank in range(8)
              for file in range(8)
              if board[rank, file] in [Piece(team, t) for t in types]])

def search_lines(board : np.array, origin : np.array, dirs : np.array, collect_all : bool = False):
    """Search along dirs until piece or edge of board is found. 
    Parameter collect_all determines whether all positions along 
    direction should be collected as well."""

    collected_poss = [[] if collect_all else None for _ in range(len(dirs))]
    collected_pieces = [None for _ in range(len(dirs))]

    dist = 1
    remaining_dirs = np.array([True for _ in range(len(dirs))])
    while remaining_dirs.any():
        poss = origin + dist * dirs[remaining_dirs]
        dist += 1

        for pos, dir_i in zip(poss, np.arange(len(dirs))[remaining_dirs]):
            if collect_all:
                collected_poss[dir_i].append(pos)
                if board[*pos] is not None:
                    collected_pieces.append(board[*pos])
                    remaining_dirs[dir_i] = False
            elif board[*pos] is not None:
                collected_poss.append(pos)
                collected_pieces[dir_i] = board[*pos]
                remaining_dirs[dir_i] = False
    
    return collected_poss, collected_pieces


