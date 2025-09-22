from game.model import Board, Piece
from game.constants import *
from game.utils import within_bounds



__all__ = ["get_king_state"]

class KingState:
    around : np.array
    check_king : bool
    check_queen : bool
    king_path : np.array
    queen_path : np.array

    def __init__(self, around, check_king, check_queen, king_path, queen_path):
        self.around = around
        self.check_king = check_king
        self.check_queen = check_queen
        self.king_path = king_path
        self.queen_path = queen_path



king_side_diffs = [[0, 2]]
queen_side_diffs = [[0, -2], [0, -3]]

around_coords = [[around_rank, around_file] for around_rank in range(3) for around_file in range(3)]

def to_real_coord(around_coord : np.array, king_coord : np.array) -> np.array:
    return king_coord + around_coord - 1


def king_control(king_state : KingState, board : Board, opponent : Team, king_coord : np.array):
    """Update king_state to account for squares controlled by opponent king."""
    def within_king_range(coord, king_coord) -> bool:
        return (np.abs(king_coord - coord) <= 1).all()

    other_king_coord = board.get_king(opponent).coord
    for around_coord in around_coords:
        if not king_state.around[*around_coord]:
            to_control = to_real_coord(around_coord, king_coord)
            king_state.around[*around_coord] = within_king_range(other_king_coord, to_control)
        
    if king_state.check_king:
        king_state.check_king = not king_state.around[1, 2] and np.any([within_king_range(coord, other_king_coord) for coord in king_state.king_path])
    
    if king_state.check_queen:
        king_state.check_queen = not king_state.around[1, 0] and np.any([within_king_range(coord, other_king_coord) for coord in king_state.queen_path])


def pawn_control(king_state : KingState, board : Board, opponent : Team, king_coord : np.array):
    pawn_direction = -1 if opponent == WHITE else 1
    def attacked_by_pawn(coord, pawn_coord) -> bool:
        return (coord == (pawn_coord + [pawn_direction, -1])).all() or (coord == (pawn_coord + [pawn_direction, 1])).all()
    
    pawns = board.team_and_type_map[opponent][PAWN]
    for around_coord in around_coords:
        if not king_state.around[*around_coord]:
            to_control = to_real_coord(around_coord, king_coord)
            king_state.around[*around_coord] = np.any([attacked_by_pawn(to_control, pawn.coord) for pawn in pawns])
        
    if king_state.check_king:
        king_state.check_king = not king_state.around[1, 2] and \
            np.any([attacked_by_pawn(coord, pawn.coord) for coord in king_state.king_path for pawn in pawns])
    
    if king_state.check_queen:
        king_state.check_queen = not king_state.around[1, 0] and \
            np.any([attacked_by_pawn(coord, pawn.coord) for coord in king_state.queen_path for pawn in pawns])


def knight_control(king_state : KingState, board : Board, opponent : Team, king_coord : np.array):
    def attacked_by_knight(coord):
        for maybe_knight in [board.coord_map[*maybe_knight_coord] for maybe_knight_coord in coord + knight_diffs if within_bounds(*maybe_knight_coord)]:
            if maybe_knight.piece_type == KNIGHT and maybe_knight.team == opponent:
                return True
        return False
    
    for around_coord in around_coords:
        if not king_state.around[*around_coord]:
            to_control = to_real_coord(around_coord, king_coord)
            king_state.around[*around_coord] = attacked_by_knight(to_control)
    
    if king_state.check_king:
        king_state.check_king = not king_state.around[1, 2] and \
            np.any([attacked_by_knight(coord) for coord in king_state.king_path])
    
    if king_state.check_queen:
        king_state.check_queen = not king_state.around[1, 0] and \
            np.any([attacked_by_knight(coord) for coord in king_state.queen_path])



def get_king_state(board : Board, team : Team):
    king_coord = board.get_king(team).coord
    to_check = [coord for coord in [king_coord + [rank_diff, file_diff] for rank_diff in range(-1, 2) for file_diff in range(-1, 2)]
                if within_bounds(*coord) and not (coord == king_coord).all()]
    
    # check if opponent controls squares for castling
    castle_path_king_side = king_coord + king_side_diffs
    castle_path_queen_side = king_coord + queen_side_diffs
    check_king = board.king_side_castle[team] and np.all([board.coord_map[*coord] is None for coord in castle_path_king_side])
    check_queen = board.queen_side_castle[team] and np.all([board.coord_map[*coord] is None for coord in castle_path_queen_side])

    king_state = KingState(np.zeros((3, 3)), check_king, check_queen, castle_path_king_side, castle_path_queen_side)

    # OPPONENT KING
    other_king_coord = board.get_king(other_team(team)).coord
    for coord in to_check:
        pass

    # OPPONENT PAWNS
    # OPPONENT KNIGHTS
    # DIAGONALS
    # LATERALS
    pass