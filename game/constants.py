import numpy as np

KING, QUEEN, ROOK, BISHOP, KNIGHT, PAWN = np.arange(6)

WHITE, BLACK = [0, 1]

knight_diffs = [[rank_diff, file_diff] 
                for rank_diff in [-2, -1, 1, 2]
                for file_diff in [-2, -1, 1, 2]
                if abs(rank_diff) != abs(file_diff)]