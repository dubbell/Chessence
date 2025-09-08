from .constants import *
from .model import Board


piece_to_files = [
    [4],
    [3],
    [0, 7],
    [1, 6],
    [2, 5]]


def within_bounds(rank : int, file : int) -> bool:
    return rank <= 7 and rank >= 0 and file <= 7 and file >= 0