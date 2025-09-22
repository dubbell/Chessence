import numpy as np
from typing import List, Tuple, Mapping
from .constants import *
from .model import Piece, Board, Move
from .king_state import get_king_state
from .utils import within_bounds


def get_moves(board : Board, team : Team, en_passant : np.array | Tuple[int, int] = None) -> Mapping[Piece, List[Move]]:
    """Returns map from pieces to a lists of moves.
       None if in checkmate, and {} if no moves are available (stalemate)."""

    # map from piece to moves
    moves : Mapping[Piece, List[Move]] = {}

    # draw from threefold repetition or 50 move rule
    if board.check_threefold() or board.check_50_move_rule():
        return moves

    # controlled squares around king, coords of pieces pinned to king, and the direction from which they are pinned
    controlled, pin_coords, pin_dirs = get_king_state(board, team)

    # lookup map of all pins, [0, 0] if no pin and direction if pin exists
    pin_map = np.zeros((8, 8, 2), dtype=int)
    for pin_coord, pin_dir in zip(pin_coords, pin_dirs):
        pin_map[*pin_coord] = pin_dir

    # squares populated by team
    team_pop = np.zeros((8, 8), dtype=int)
    for piece in board.of_team(team):
        team_pop[*piece.coord] = 1

    # squares populated by opponent
    oppo_pop = np.zeros((8, 8), dtype=int)
    for piece in board.of_team(other_team(team)):
        oppo_pop[*piece.coord] = 1
    

    # king moves
    king_piece = board.get_king(team)
    if king_piece is not None:
        king_moves = []
        king_rank, king_file = king_piece.coord
        for controlled_rank in range(3):
            for controlled_file in range(3):
                #king's square
                if controlled_rank == 1 and controlled_file == 1:
                    continue
                to_rank, to_file = king_rank + controlled_rank - 1, king_file + controlled_file - 1
                if within_bounds(to_rank, to_file) \
                        and controlled[controlled_rank, controlled_file] == 0 \
                        and team_pop[to_rank, to_file] == 0:
                    king_moves.append(Move([to_rank, to_file]))
        
        if len(king_moves) > 0:
            moves[king_piece] = king_moves

    # if king is in check
    if controlled[1, 1] == 1:
        return moves if len(moves) > 0 else None  # None if checkmate
    
    # pawn moves
    for pawn_piece in board.of_team_and_type(team, PAWN):
        pawn_moves = []

        pawn_coord = pawn_piece.coord
        pin_dir = pin_map[*pawn_coord]
        rank_diff = 1 if team == BLACK else -1

        # forward moves
        if pin_map[*pawn_coord, 1] == 0:  # pinn not from directly front or back
            if oppo_pop[*(pawn_coord + [rank_diff, 0])] == 0:
                pawn_moves.append(Move(pawn_coord + [rank_diff, 0]))
            
                if (pawn_coord[0] == 1 and team == BLACK or pawn_coord[0] == 6 and team == WHITE) \
                        and oppo_pop[*(pawn_coord + [rank_diff * 2, 0])] == 0:
                    pawn_moves.append(Move(pawn_coord + [rank_diff * 2, 0]))


        # capture moves
        for file_diff in [-1, 1]:
            # in bounds of board
            if not within_bounds(*(pawn_coord + [rank_diff, file_diff])):
                continue
            
            # whether pawn is pinned from relevant direction (in relation to move direction)
            if not (pin_dir == 0).all() \
                    and not (pin_dir == [rank_diff, file_diff]).all() \
                    and not (-pin_dir == [rank_diff, file_diff]).all():
                continue
            
            # if square occupied by opponent, then the move is possible
            if oppo_pop[*(pawn_coord + [rank_diff, file_diff])] == 1:
                pawn_moves.append(Move(pawn_coord + [rank_diff, file_diff]))

            # in case of en passant
            elif en_passant is not None and (en_passant == pawn_coord + [0, file_diff]).all():
                # if en passant pawn is pinned, then ignore
                if not (pin_map[*en_passant] == 0).all():
                    continue
                
                # rare case where opposite team pawns next to each other are both laterally pinned to a king
                elif king_piece is not None and king_piece.coord[0] == (3 if team == WHITE else 4):
                    check_direction = -1 if king_piece.coord[1] > pawn_coord[1] else 1
                    check_file = king_piece.coord[1] + check_direction

                    rare_double_pawn_pin = False
                    
                    # check squares on side of pawns opposite of king until piece or edge is found
                    while within_bounds(pawn_coord[0], check_file):
                        # ignore the two pawns
                        if check_file == pawn_coord[1] or check_file == pawn_coord[1] + file_diff:
                            check_file += check_direction
                            continue
                        
                        # check coordinate for piece
                        check_piece = board.coord_map.get((pawn_coord[0], check_file))
                        
                        # if piece found
                        if check_piece is not None:
                            # rare double pawn pin check
                            if check_piece.team == other_team(team) and check_piece.piece_type in [QUEEN, ROOK]:
                                rare_double_pawn_pin = True
                            break
                        
                        check_file += check_direction
                    
                    # no rare double pawn pin, then add en passant move
                    if not rare_double_pawn_pin:
                        pawn_moves.append(Move(pawn_coord + [rank_diff, file_diff]))
                
                # no pins, add en passant move
                else:
                    pawn_moves.append(Move(pawn_coord + [rank_diff, file_diff]))

        if len(pawn_moves) > 0:
            moves[pawn_piece] = pawn_moves


    # knight moves
    for knight_piece in board.of_team_and_type(team, KNIGHT):
        knight_moves = []

        knight_coord = knight_piece.coord
        # if not pinned, then add moves
        if (pin_map[*knight_coord] == 0).all():
            knight_moves.extend([Move(to_coord) 
                                 for to_coord in knight_coord + knight_diffs
                                 if within_bounds(*to_coord)  # bounds checking
                                 and team_pop[*to_coord] == 0])  # not populated by team
        
        if len(knight_moves) > 0:
            moves[knight_piece] = knight_moves
    

    # bishop moves
    for bishop_piece in board.of_team_and_type(team, BISHOP):
        bishop_moves = []

        bishop_coord = bishop_piece.coord

        # determine directions in which bishop can move
        if (pin_map[*bishop_coord] == 0).all():
            bishop_dirs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        elif (pin_map[*bishop_coord] == [1, 1]).all() or (pin_map[*bishop_coord] == [-1, -1]).all():
            bishop_dirs = np.array([[-1, -1], [1, 1]])
        elif (pin_map[*bishop_coord] == [-1, 1]).all() or (pin_map[*bishop_coord] == [1, -1]).all():
            bishop_dirs = np.array([[1, -1], [-1, 1]])
        else:
            continue  # can't move if pinned from lateral direction

        bishop_moves.extend([
            Move(to_coord)
            for to_coord in get_moves_along_directions(bishop_coord, bishop_dirs, team_pop, oppo_pop)])
    
        if len(bishop_moves) > 0:
            moves[bishop_piece] = bishop_moves
    

    # rook moves
    for rook_piece in board.of_team_and_type(team, ROOK):
        rook_moves = []

        rook_coord = rook_piece.coord

        # determine directions in which rook can move
        if (pin_map[*rook_coord] == 0).all():
            rook_dirs = np.array([[-1, 0], [0, 1], [0, -1], [1, 0]])
        elif (pin_map[*rook_coord] == [0, 1]).all() or (pin_map[*rook_coord] == [0, -1]).all():
            rook_dirs = np.array([[0, -1], [0, 1]])
        elif (pin_map[*rook_coord] == [-1, 0]).all() or (pin_map[*rook_coord] == [1, 0]).all():
            rook_dirs = np.array([[1, 0], [-1, 0]])
        else:
            continue  # can't move if pinned from diagonal direction
    
        rook_moves.extend([
            Move(to_coord)
            for to_coord in get_moves_along_directions(rook_coord, rook_dirs, team_pop, oppo_pop)])

        if len(rook_moves) > 0:
            moves[rook_piece] = rook_moves
    

    # queen moves
    for queen_piece in board.of_team_and_type(team, QUEEN):
        queen_moves = []

        queen_coord = queen_piece.coord

        if (pin_map[*queen_coord] == 0).all():
            queen_dirs = np.array([[-1, 0], [0, 1], [0, -1], [1, 0], [-1, -1], [-1, 1], [1, -1], [1, 1]])
        else:
            queen_dirs = np.array([pin_map[*queen_coord], -pin_map[*queen_coord]])
        
        queen_moves.extend([
            Move(to_coord)
            for to_coord in get_moves_along_directions(queen_coord, queen_dirs, team_pop, oppo_pop)])

        if len(queen_moves) > 0:
            moves[queen_piece] = queen_moves


    return moves


def get_moves_along_directions(origin : np.array, dirs : np.array, team_pop : np.array, oppo_pop : np.array):
    """Gets all moves for piece that can move along given directions."""

    moves = []

    for dist in range(1, 8):
        if len(dirs) == 0:
            break

        to_coords = origin + dirs * dist
        remove = []

        for dir_index, to_coord in enumerate(to_coords):
            if not within_bounds(*to_coord) or team_pop[*to_coord]:
                remove.append(dir_index)
            elif oppo_pop[*to_coord]:
                moves.append(to_coord)
                remove.append(dir_index)
            else:
                moves.append(to_coord)
        
        dirs = np.delete(dirs, remove, axis=0)
    
    return moves