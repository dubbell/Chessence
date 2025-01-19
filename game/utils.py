import numpy as np
from model import *
from typing import Union, Tuple


def get_starting_board():
    pieces = { team : { Type(t_i) : [] for t_i in range(1, 6) } for team in [Team.WHITE, Team.BLACK]}
    board = [[None for _ in range(8)] for _ in range(8)]
    for rank in range(8):
        for file in range(8):
            piece = Piece(Team.BLACK, Type.ROOK, rank, file) if (rank, file) in [(0, 0), (0, 7)] else \
                    Piece(Team.BLACK, Type.KNIGHT, rank, file) if (rank, file) in [(0, 1), (0, 6)] else \
                    Piece(Team.BLACK, Type.BISHOP, rank, file) if (rank, file) in [(0, 2), (0, 5)] else \
                    Piece(Team.BLACK, Type.QUEEN, rank, file) if (rank, file) == (0, 3) else \
                    Piece(Team.BLACK, Type.BISHOP, rank, file) if (rank, file) == (0, 4) else \
                    Piece(Team.BLACK, Type.PAWN, rank, file) if rank == 1 else \
                    Piece(Team.WHITE, Type.ROOK, rank, file) if (rank, file) in [(7, 0), (7, 7)] else \
                    Piece(Team.WHITE, Type.KNIGHT, rank, file) if (rank, file) in [(7, 1), (7, 6)] else \
                    Piece(Team.WHITE, Type.BISHOP, rank, file) if (rank, file) in [(7, 2), (7, 5)] else \
                    Piece(Team.WHITE, Type.QUEEN, rank, file) if (rank, file) == (7, 3) else \
                    Piece(Team.WHITE, Type.BISHOP, rank, file) if (rank, file) == (7, 4) else \
                    Piece(Team.WHITE, Type.PAWN, rank, file) if rank == 6 else \
                    None
            
            if piece is not None:
                if piece.type == Type.KING:
                    pieces[piece.team][Type.KING] = piece
                else:
                    pieces[piece.team][piece.type].append(piece)
            board[rank][file] = piece
        
    return pieces, np.array(board)


def print_board(board : np.array):
    print(np.array([["  " if square is None else square for square in rank] for rank in board]))


def within_bounds(*args : Union[Tuple[int], Tuple[np.array]]) -> bool:
    """True if position is within the bounds of the chess board."""
    rank, file = args[0] if len(args) == 1 else args
    return rank >= 0 and rank <= 7 and file >= 0 and file <= 7

def out_of_bounds(*args : Union[Tuple[int], Tuple[np.array]]) -> bool:
    """Complement of within_bounds."""
    return not within_bounds(args)

def opponent(team : Team):
    return Team.WHITE if team == Team.BLACK else Team.BLACK


pieces, board = get_starting_board()

print(pieces)
print(board.shape)
