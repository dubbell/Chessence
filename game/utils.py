from constants import *
from model import Board


piece_to_files = [
    [4],
    [3],
    [0, 7],
    [1, 6],
    [2, 5]]


def get_starting_board() -> Board:

    board = Board()
    
    for team, piece_rank, pawn_rank in zip([WHITE, BLACK], [7, 0], [6, 1]):
        board.coords[team] = np.zeros((16, 2), dtype=int)
        board.types[team] = np.zeros(16, dtype=int)
        count = 0
        for piece_type, files in zip([KING, QUEEN, ROOK, BISHOP, KNIGHT], piece_to_files):
            board.type_locs[team, piece_type] = count
            for file in files:
                board.coords[team][count] = [piece_rank, file]
                board.types[team][count] = piece_type
                count += 1
        
        board.coords[team][count:] = [[pawn_rank, file] for file in range(8)]
        board.types[team][count:] = PAWN
        board.type_locs[team, PAWN] = count

    return board

