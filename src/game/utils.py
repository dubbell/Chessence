from game.constants import *
import numpy as np


piece_to_files = [
    [4],
    [3],
    [0, 7],
    [1, 6],
    [2, 5]]


def within_bounds(rank : int, file : int) -> bool:
    return rank <= 7 and rank >= 0 and file <= 7 and file >= 0

def to_index(coord : np.array):
    return np.ravel_multi_index(coord, (8, 8))