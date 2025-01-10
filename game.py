import numpy as np
from collections.abc import Callable


WHITE = 0
BLACK = 1

TEAM_SLICES = [slice(0, 6), slice(6, None)]

PAWN = 0
KING = 1
QUEEN = 2
ROOK = 3
KNIGHT = 4
BISHOP = 5



    
def locate(sub_board : np.array) -> np.array:
    """Returns list of coordinates for each piece found on the (sub-)board."""
    board_sum = sub_board if len(sub_board.shape) == 2 else sub_board.sum(axis=0)
    flat_indices = np.arange(64)[board_sum.flatten() > 0]
    return np.array(np.unravel_index(flat_indices, (8, 8))).T

def locate_first(piece_states : np.array) -> np.array:
    """Faster version of locate. Only takes a single piece's state and looks for first instance. Used for kings."""
    for rank in range(8):
        for file in range(8):
            if piece_states[rank, file] > 0:
                return np.array([rank, file])


def within_bounds(pos : np.array):
    """True if position is within the bounds of the chess board."""
    return (pos >= 0).all() and (pos <= 7).all()

def out_of_bounds(pos : np.array):
    """Complement of within_bounds."""
    return not within_bounds(pos)
    

class Game:

    def get_starting_board(self) -> np.array:
        """Returns board with pieces on starting positions."""

        board = np.zeros((12, 8, 8))
        white_board = board[:6, :, :]
        black_board = board[6:, :, :]

        white_board[PAWN, 6, :] = 1
        white_board[KING, 7, 4] = 1
        white_board[QUEEN, 7, 3] = 1
        white_board[ROOK, 7, [0, 7]] = 1
        white_board[KNIGHT, 7, [1, 6]] = 1
        white_board[BISHOP, 7, [2, 5]] = 1
        
        black_board[PAWN, 1, :] = 1
        black_board[KING, 0, 4] = 1
        black_board[QUEEN, 0, 3] = 1
        black_board[ROOK, 0, [0, 7]] = 1
        black_board[KNIGHT, 0, [1, 6]] = 1
        black_board[BISHOP, 0, [2, 5]] = 1

        return board, white_board, black_board
    

    def __init__(self):

        self.board, self.white_board, self.black_board = self.get_starting_board()

    
    def cast_rays(self, pos : np.array, dirs : np.array, search_condition : Callable[[np.array], bool] = None):
        """Search from given position by casting rays in the directions given by dirs. search_condition is function
           that specifies what is being searched for. If not None, then cast_ray returns True when an instance is found.
           If search_condition is None, then cast_ray returns all pieces that are encountered when searching dirs."""
        pass


    def is_controlled_by(self, board : np.array, pos : np.array, team : int):
        pass

    
    def get_moves(self, board : np.array, team : int):

        team_board = board[TEAM_SLICES[team]]
        opp_board = board[TEAM_SLICES[int(not team)]]

        team_board_populated = team_board.sum(axis=0) > 0
        opp_board_populated = opp_board.sum(axis=0) > 0

        available_moves = []  # to be computed
        
        # king moves
        king_pos = locate_first(team_board[KING])
        for pos_diff in [np.array([rank, file]) for rank in [0, -1, 1] for file in [0, -1, 1]]:
            new_pos = new_rank, new_file = king_pos + pos_diff
            
            if out_of_bounds(new_pos) or team_board_populated[new_rank, new_file] or self.is_controlled_by(board, new_pos, not team):
                continue
            else:
                available_moves.append(Move(king_pos, new_pos, KING))
        
        # pawn moves
        pawn_poss = locate(team_board[PAWN])
        for pawn_pos in pawn_poss:
            move_dir = 1 if team == BLACK else -1

            # candidate move directions
            candidate_diffs = np.array([[move_dir, -1], [move_dir, 0], [move_dir, 1]])
            # candidate destination positions
            candidate_poss = pawn_pos + candidate_diffs

            to_delete = []

            # disregard first diagonal direction if no opponent at position
            diag_rank1, diag_file1 = candidate_poss[0]
            if not opp_board_populated[diag_rank1, diag_file1]:
                to_delete.append(0)

            # disregard moving forward if position already populated
            lat_rank, lat_file = candidate_poss[1]
            if opp_board_populated[lat_rank, lat_file] or team_board_populated[lat_rank, lat_file]:
                to_delete.append(1)
            
            # disregard second diagonal direction if no opponent at position
            diag_rank2, diag_file2 = candidate_poss[2]
            if not opp_board_populated[diag_rank2, diag_file2]:
                to_delete.append(2)
            
            # Delete disregarded positions and directions. If empty, move on to next pawn.
            candidate_diffs = np.delete(candidate_diffs, to_delete, axis=0)
            candidate_poss = np.delete(candidate_poss, to_delete, axis=0)
            if len(candidate_poss) == 0:
                continue

            # now checking for pins

            # one_dirs is directions from which a pin can exist, lateral and diagonal
            one_dirs = np.array([[0, -1], [-1, -1], [-1, 0], [-1, 1]])
            # opposite directions of the ones in one_dirs
            two_dirs = -one_dirs

            # results of searching in one_dirs and two_dirs
            one_casts = self.cast_rays(pawn_pos, one_dirs)
            two_casts = self.cast_rays(pawn_pos, two_dirs)
            
            for candidate_pos, candidate_diff in zip(candidate_poss, candidate_diffs):
                relevant_casts = np.logical_not(
                    np.logical_or(
                        (candidate_diff == one_dirs).all(axis=1), 
                        (candidate_diff == two_dirs).all(axis=1)))
            






class Move:
    def __init__(self, from_pos : np.array, to_pos : np.array, piece_type : int):
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.piece_type = piece_type

