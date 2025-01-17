import numpy as np
from search import search_lines, locate, locate_first
from model import Team, Type
from constants import ALL_DIRS



def get_moves(board : np.array, team : int):
    king_pos = locate_first(board, team, Type.KING)

    king_found_poss, king_found_pieces = search_lines(board, king_pos, ALL_DIRS)

    
