import numpy as np
from search import cast_rays, locate, locate_first
from model import Team, Type
from constants import ALL_DIRS
from utils import opponent



def get_moves(board : np.array, team : Team):
    king_pos = locate_first(board, team, Type.KING)

    opponent_control = np.zeros((8, 8))

    # calculate pins
    pinned = []
    king_covered, king_hits = cast_rays(board, king_pos, ALL_DIRS)
    opposite = [4, 5, 6, 7, 0, 1, 2, 3]

    for pin_pos, pin_piece in zip(*locate(board, opponent(team), [Type.QUEEN, Type.ROOK, Type.BISHOP])):
        if pin_piece.type == Type.QUEEN:
            # cast rays in all directions
            pin_covered, pin_hits = cast_rays(board, pin_pos, ALL_DIRS, True)
            # positions where rays hit
            pin_hit_poss = []

            # determine squares controlled by opponent
            for line_poss, pin_hit in zip(pin_covered, pin_hits):
                # all covered positions are controlled by opponent
                for covered_pos in line_poss:
                    opponent_control[*covered_pos] = 1
                pin_hit_poss.append(pin_covered[-1])

            # pin checking
            # if king ray hits the same target as a queen ray and they are from opposite sides, then it is a pin
            for dir_i, (king_hit_pos, king_hit, pin_hit_pos, pin_hit) in zip(king_covered, king_hits, pin_hit_poss[opposite], pin_hits[opposite]):
                if king_hit == pin_hit and (king_hit_pos == pin_hit_pos).all():
                    pinned.append(king_hit)
                    king_hit.pins.append(ALL_DIRS[dir_i])

        if pin_piece.type == Type.BISHOP:
            
    

    
    for piece in pinned:
        piece.pins = []



