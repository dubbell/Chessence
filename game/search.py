import numpy as np
from model import Team, Type, Piece
from typing import List, Tuple, Union


def locate_first(board : np.array, team : Team, type : Type) -> np.array:
    for rank in range(8):
        for file in range(8):
            if board[rank, file] == Piece(team, type):
                return np.array([rank, file])

def locate(board : np.array, team : Team, types : List[Type]) -> Tuple[List[np.array], List[Piece]]:
    poss = []
    pieces = []
    for rank in range(8):
        for  file in range(8):
            if board[rank, file] in [Piece(team, t) for t in types]:
                poss.append(np.array([rank, file]))
                pieces.append(board[rank, file])
    
    return poss, pieces

def cast_rays(board : np.array, 
              origin : np.array, 
              dirs : np.array, 
              collect_all : bool = False
    ) -> Tuple[Union[List[np.array], List[List[np.array]]], List[Piece]]:
    """Search along dirs until piece or edge of board is found. 
    Parameter collect_all determines whether all positions along 
    direction should be collected as well."""

    # Surface covered by each ray. If collect_all is False, then only the position of the hit is returned.
    covered = [[] if collect_all else None for _ in range(len(dirs))]
    # What each ray hits, either None or a piece.
    hits = [None for _ in range(len(dirs))]

    dist = 1
    remaining_dirs = np.array([True for _ in range(len(dirs))])
    while remaining_dirs.any():
        poss = origin + dist * dirs[remaining_dirs]
        dist += 1

        for pos, dir_i in zip(poss, np.arange(len(dirs))[remaining_dirs]):
            if collect_all:
                covered[dir_i].append(pos)
                if board[*pos] is not None:
                    hits.append(board[*pos])
                    remaining_dirs[dir_i] = False
            elif board[*pos] is not None:
                covered.append(pos)
                hits[dir_i] = board[*pos]
                remaining_dirs[dir_i] = False
    
    return covered, hits


