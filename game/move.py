import numpy as np
from search import cast_rays, locate, locate_first, locate_all
from model import Team, Type
from constants import ALL_DIRS, DIAG_DIRS, LAT_DIRS
from utils import opponent



def get_moves(board : np.array, pieces : map, team : Team):

    opponent_control = np.zeros((8, 8))

    king_coord = pieces[team][Type.KING].coord

    # calculate pins
    pinned = []
    king_covered, king_hits = cast_rays(board, king_coord, ALL_DIRS)

    # calculate pins and opponent control by queen, rooks and bishops
    for pinner_piece in [piece 
                         for piece_list in [pieces[opponent(team)][type] for type in [Type.QUEEN, Type.ROOK, Type.BISHOP]] 
                         for piece in piece_list]:
        # determine in which directions to cast rays
        if pinner_piece.type == Type.QUEEN:
            pinner_dirs = ALL_DIRS
            opposite_dir_indices = [4, 5, 6, 7, 0, 1, 2, 3]
        else:
            if pinner_piece.type == Type.BISHOP:
                pinner_dirs = DIAG_DIRS
            else:
                pinner_dirs = LAT_DIRS
            opposite_dir_indices = [2, 3, 0, 1]
        
        # cast rays in directions
        pinner_covered, pinner_hits = cast_rays(board, pinner_piece.coord, pinner_dirs, True)
        # positions where rays hit
        pin_hit_coords = []
        
        # determine opponent control and what the rays hit
        for line_coords, pin_hit in zip(pinner_covered, pinner_hits):
            for covered_coord in line_coords:
                opponent_control[*covered_coord] = 1
            pin_hit_coords.append(pinner_covered[-1] if pin_hit is not None else None)

        # pin checking
        # if king ray hits the same target as a piece ray and they are from opposite sides, then it is a pin
        for dir_i, king_hit_coord, king_hit, pin_hit_coord, pin_hit in \
                zip(np.arange(len(pinner_hits)), king_covered, king_hits, pin_hit_coords[opposite_dir_indices], pinner_hits[opposite_dir_indices]):
            if king_hit == pin_hit and (king_hit_coord == pin_hit_coord).all():
                pinned.append(king_hit)
                king_hit.pins.append(ALL_DIRS[dir_i])
    





    
    for piece in pinned:
        piece.pins = []



