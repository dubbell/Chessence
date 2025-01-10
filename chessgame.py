import numpy as np
import itertools
from typing import List


WHITE = 0
BLACK = 1
DRAW = 2
IN_PROGRESS = 3

KING = 0
QUEEN = 1
ROOK = 2
KNIGHT = 3
BISHOP = 4
PAWN = 5

TEAM = 0
TYPE = 1
RANK = 2
FILE = 3
POS = slice(2, 4)


DIAGONAL_DIRS = np.array(
    [(rank_diff, file_diff)
     for rank_diff in [1, -1]
     for file_diff in [1, -1]])

LATERAL_DIRS = np.array(
    [(rank_diff, file_diff) 
     for rank_diff, file_diff in zip([0, -1, 1, 0], [-1, 0, 0, 1])])

KNIGHT_DIFFS = np.array(
    [(rank_diff * rank_sign, file_diff * file_sign)
     for rank_diff, file_diff in [(1, 2), (2, 1)]
     for rank_sign in [-1, 1]
     for file_sign in [-1, 1]])


PIECE_NAMES = ["KING", "QUEEN", "ROOK", "KNIGHT", "BISHOP", "PAWN"]
TEAM_NAMES = ["WHITE", "BLACK"]

def print_piece(piece : np.array):
    if piece is None:
        print(None)
    else:
        print(TEAM_NAMES[piece[TEAM]], PIECE_NAMES[piece[TYPE]], piece[POS])

def print_board(board : np.array):
    print(np.array([[piece_char(piece) for piece in rank] for rank in board]))

def piece_char(piece : np.array):
    return (" " if piece is None else
            "K" if piece[TYPE] == KING else
            "Q" if piece[TYPE] == QUEEN else
            "R" if piece[TYPE] == ROOK else
            "N" if piece[TYPE] == KNIGHT else 
            "B" if piece[TYPE] == BISHOP else
            "P")

def within_bounds(pos : np.array):
    return pos[0] >= 0 and pos[0] <= 7 and pos[1] >= 0 and pos[1] <= 7
def within_bounds(rank : int, file : int):
    return rank >= 0 and rank <= 7 and file >= 0 and file <= 7

def other_team(team : int):
    return 1 if team == 0 else 0

    

class ChessGame:
    
    def __init__(self):
        self.status = IN_PROGRESS
        
        # board setup
        self.board = [[] for _ in range(8)]

        black_pieces = [[] for _ in range(6)]
        white_pieces = [[] for _ in range(6)]

        for index, piece_type in enumerate([ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK]):
            
            black_piece = np.array([BLACK, piece_type, 0, index])
            self.board[0].append(black_piece)
            black_pieces[piece_type].append(black_piece)

            black_pawn = np.array([BLACK, PAWN, 1, index])
            self.board[1].append(black_pawn)
            black_pieces[PAWN].append(black_pawn)

            white_piece = np.array([WHITE, piece_type, 0, index])
            self.board[7].append(white_piece)
            white_pieces[piece_type].append(white_piece)

            white_pawn = np.array([WHITE, PAWN, 6, index])
            self.board[6].append(white_pawn)
            white_pieces[PAWN].append(white_pawn)

            for rank in range(2, 6):
                self.board[rank].append(None)

        self.pieces = [black_pieces, white_pieces]
        self.board = np.array([piece for rank in self.board for piece in rank], dtype = object).reshape((8, 8))

        # game states
        self.turn = WHITE
        self.king_move = [False, False]


    def search_directions(self, pos : np.array, directions : np.array, team : int = None, types : List[int] = None):
        """Search for `types` in the specified `directions`, blocked by pieces. If team and types are unspecified, 
           then searches in all `directions` and returns the pieces that it finds. If they are specified, then the
           direction from which the piece was found is returned.
           
           Returns: direction / None (for searching specific piece)
                    or
                    pieces (for searching all given directions)"""
        
        # located pieces
        located = [None for _ in range(len(directions))]

        if team != None and types != None:
            exit_condition = lambda piece: piece[TEAM] == team and piece[TYPE] in types
        elif team != None:
            exit_condition = lambda piece: piece[TEAM] == team
        elif types != None:
            exit_condition = lambda piece: piece[TYPE] in types
        else:
            exit_condition = None

        # remaining directions to continue checking
        remaining = np.array([True for _ in range(len(directions))])
        
        for dist in range(1, 8):
            if not remaining.any():
                break

            positions = pos + directions[remaining] * dist

            for pos_i, (position_rank, position_file) in zip(np.arange(len(directions))[remaining], positions):
                # out of bounds
                if not within_bounds(position_rank, position_file):
                    remaining[pos_i] = False  # remove direction
                    continue

                piece = self.board[position_rank, position_file]
                # no piece at position, continue
                if piece is None:  
                    continue
                # piece of given team and type
                elif exit_condition != None and exit_condition(piece):
                    return True
                # piece blocks
                else:
                    located[pos_i] = piece
                    remaining[pos_i] = False  # remove direction

        return False if exit_condition != None else located


    def is_controlled_by(self, pos : np.array, team : int):
        """True if given square is being controlled by given team."""

        rank, file = pos
        pieces = self.pieces[team]
        
        # king
        if abs(pieces[KING][0][RANK] - rank) <= 1 and abs(pieces[KING][0][FILE] - file) <= 1:
            return True
        
        # pawns
        pawn_threats = [[rank - 1 + team * 2, file - 1], [rank - 1 + team * 2, file + 1]]
        for pawn in pieces[PAWN]:
            if pawn[POS] in pawn_threats:
                return True
        
        # knights
        knight_threats = filter(within_bounds, pos + KNIGHT_DIFFS)
        for knight in pieces[KNIGHT]:
            if knight[POS] in knight_threats:
                return True
        
        return (self.search_directions(pos, team, DIAGONAL_DIRS, [BISHOP, QUEEN] is not None) 
             or self.search_directions(pos, team, LATERAL_DIRS, [ROOK, QUEEN]) is not None)
    

    def find_pin(self, piece : np.array, king_team : int):
        """Checks if piece is pinned, and returns the direction of the responsible piece. Returns None otherwise."""
        pos = piece[POS]
        
        directions = np.vstack((DIAGONAL_DIRS, LATERAL_DIRS))
        visible_pieces = self.search_directions(pos, directions)
        for piece_index, piece in enumerate(visible_pieces):
            if piece[TYPE] == KING and piece[TEAM] == king_team:
                opposite_index = (piece_index + 
                    3 if piece_index in [0, 4] else
                    1 if piece_index in [1, 5] else
                   -1 if piece_index in [2, 6] else -3)

                opposite = visible_pieces[opposite_index]
                
                if opposite is None:
                    return None
                elif (opposite[TEAM] == other_team(king_team) and (
                           opposite[TYPE] in [BISHOP, QUEEN] and opposite_index <= 3
                        or opposite[TYPE] in [ROOK, QUEEN] and opposite_index >= 4)):
                    
                    return directions[opposite_index]
                else:
                    return None

        return None



    def get_moves(self):
        pass

        

    def load_game(self, game_string : str, status):
        self.moves = game_string.split(" ")
        for i in range(0, len(self.moves), 2):
            self.moves[i] = self.moves[i][self.moves[i].index(".") + 1:]
        
        self.status = status


print(LATERAL_DIRS)
print(DIAGONAL_DIRS)