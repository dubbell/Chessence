from game.model import Board, Move, Piece
from game.constants import *
from game.utils import within_bounds, to_index
from game.move_calc import get_moves

import numpy as np
from typing import List

import pytest


def move_matrix_contains_exactly(move_matrix : np.array, moves : List[Move]):
    true_matrix = np.zeros((64, 64))
    for move in moves:
        select = to_index(move.piece.coord)
        target = to_index(move.to_coord)
        assert move_matrix[select, target], f"move {move.piece} to {move.to_coord} not in move matrix"
        true_matrix[select, target] = 1
    
    for select, target in [(select, target) for select in range(64) for target in range(64)]:
        assert true_matrix[select, target] == move_matrix[select, target], \
            f"extra move in move_matrix: {np.unravel_index(select, (8, 8))} to {np.unravel_index(target, (8, 8))}"


def piece_can_move_exactly(piece : Piece, move_matrix : np.array, moves : List[Move]):
    true_matrix = np.zeros((64, 64))
    select = to_index(piece.coord)
    for move in moves:
        target = to_index(move.to_coord)
        assert move_matrix[select, target], f"move {move.piece} to {move.to_coord} not in move matrix"
        true_matrix[select, target] = 1
    
    for target in range(64):
        assert true_matrix[select, target] == move_matrix[select, target], \
            f"extra move in move_matrix: {np.unravel_index(select, (8, 8))} to {np.unravel_index(target, (8, 8))}"


@pytest.mark.parametrize("king_coord", [np.array([rank, file]) for rank in range(8) for file in range(8)])
def test_king_moves(king_coord):
    board = Board()
    king_piece = board.add_piece(KING, WHITE, *king_coord)

    move_matrix = get_moves(board, WHITE)

    board_state = board.get_state(WHITE)

    true_moves = [Move(king_piece, move) for move in king_coord + [
            [rank_diff, file_diff]
            for rank_diff in [-1, 0, 1]
            for file_diff in [-1, 0, 1]
            if rank_diff != 0 or file_diff != 0]
        if within_bounds(*move)]

    assert (board_state == board.get_state(WHITE)).all(), "board state not reversed"

    move_matrix_contains_exactly(move_matrix, true_moves)


def test_king_moves_controlled():
    board = Board()
    king_coord = np.array([4, 4])
    king_piece = board.add_piece(KING, WHITE, *king_coord)

    board.add_piece(ROOK, BLACK, 3, 0)

    move_matrix = get_moves(board, WHITE)

    true_moves = [Move(king_piece, move) for move in king_coord + [
                [rank_diff, file_diff]
                for rank_diff in [-1, 0, 1]
                for file_diff in [-1, 0, 1]
                if (rank_diff != 0 or file_diff != 0) and rank_diff != -1]]
    
    move_matrix_contains_exactly(move_matrix, true_moves)

    board.add_piece(BISHOP, BLACK, 2, 3)

    move_matrix = get_moves(board, WHITE)

    true_moves.remove(Move(king_piece, [4, 5]))

    move_matrix_contains_exactly(move_matrix, true_moves)

    # pawn can move
    pawn_piece = board.add_piece(PAWN, WHITE, 4, 7)
    
    move_matrix = get_moves(board, WHITE)

    true_moves.append(Move(pawn_piece, [3, 7]))

    move_matrix_contains_exactly(move_matrix, true_moves)

    # queen checks king, pawn can no longer move
    board.add_piece(QUEEN, BLACK, 2, 2)

    move_matrix = get_moves(board, WHITE)

    true_moves.remove(Move(king_piece, [5, 5]))
    true_moves.remove(Move(pawn_piece, [3, 7]))

    move_matrix_contains_exactly(move_matrix, true_moves)

    board.add_piece(KNIGHT, BLACK, 6, 2)

    move_matrix = get_moves(board, WHITE)

    true_moves.remove(Move(king_piece, [5, 4]))
    true_moves.remove(Move(king_piece, [4, 3]))

    move_matrix_contains_exactly(move_matrix, true_moves)

    board.add_piece(KING, BLACK, 6, 4)

    move_matrix = get_moves(board, WHITE)

    assert move_matrix is None  # checkmate

    board.remove_piece_at(2, 2)  # remove checking piece
    board.remove_piece_at(4, 7)  # remove white pawn

    move_matrix = get_moves(board, WHITE)

    assert (move_matrix == 0).all()  # stalemate


def test_king_moves_blocked():
    board = Board()
    king_coord = np.array([4, 4])
    king_piece = board.add_piece(KING, WHITE, *king_coord)

    board.add_piece(PAWN, WHITE, 4, 3)

    move_matrix = get_moves(board, WHITE)

    true_moves = [Move(king_piece, move) for move in king_coord + [
            [rank_diff, file_diff]
            for rank_diff in [-1, 0, 1]
            for file_diff in [-1, 0, 1]
            if (rank_diff != 0 or file_diff != 0)]]
    
    true_moves.remove(Move(king_piece, [4, 3]))

    piece_can_move_exactly(king_piece, move_matrix, true_moves)

    board.add_piece(PAWN, WHITE, 5, 5)

    move_matrix = get_moves(board, WHITE)

    true_moves.remove(Move(king_piece, [5, 5]))

    piece_can_move_exactly(king_piece, move_matrix, true_moves)

    board.add_piece(BISHOP, BLACK, 5, 4)

    move_matrix = get_moves(board, WHITE)

    true_moves.remove(Move(king_piece, [4, 5]))  # square controlled by bishop

    piece_can_move_exactly(king_piece, move_matrix, true_moves)


@pytest.mark.parametrize("rank,team", [(7, WHITE), (0, BLACK)])
def test_castling(rank, team):
    board = Board()
    board.reset()

    assert board.king_side_castle[team]
    assert board.queen_side_castle[team]
    
    king_castle_move = np.ravel_multi_index((rank, 4), (8, 8)), np.ravel_multi_index((rank, 6), (8, 8))
    queen_castle_move = np.ravel_multi_index((rank, 4), (8, 8)), np.ravel_multi_index((rank, 2), (8, 8))

    def assert_castling(king_castle, queen_castle):
        move_matrix = get_moves(board, team)
        assert bool(move_matrix[*king_castle_move]) == king_castle, f"{king_castle} != {bool(move_matrix[*king_castle_move])}"
        assert bool(move_matrix[*queen_castle_move]) == queen_castle, f"{king_castle} != {bool(move_matrix[*queen_castle_move])}"

    assert_castling(False, False)

    board.remove_piece_at(rank, 5)
    assert_castling(False, False)

    board.remove_piece_at(rank, 6)
    assert_castling(True, False)

    board.king_side_castle[team] = False
    assert_castling(False, False)

    for file in range(3, 1, -1):
        board.remove_piece_at(rank, file)
        assert_castling(False, False)

    board.remove_piece_at(rank, 1)
    assert_castling(False, True)

    board.queen_side_castle[team] = False
    assert_castling(False, False)

    board.queen_side_castle[team] = True
    board.add_piece(KNIGHT, other_team(team), 5 if team == WHITE else 2, 2)

    assert_castling(False, False)


@pytest.mark.parametrize("team,ranks,file", [(*team_ranks, file) for team_ranks in list(zip([WHITE, BLACK], [[6, 5, 4], [1, 2, 3]])) for file in range(8)])
def test_pawn_moves(team, ranks, file):
    board = Board() 

    pawn_piece = board.add_piece(PAWN, team, ranks[0], file)
    move_matrix = get_moves(board, team)
    true_moves = [Move(pawn_piece, [rank, file]) for rank in ranks[1:]]
    move_matrix_contains_exactly(move_matrix, true_moves)

    board.add_piece(PAWN, other_team(team), ranks[2], file)
    true_moves.remove(Move(pawn_piece, [ranks[2], file]))
    move_matrix = get_moves(board, team)
    move_matrix_contains_exactly(move_matrix, true_moves)

    board.add_piece(PAWN, other_team(team), ranks[1], file)
    move_matrix = get_moves(board, team)
    assert (move_matrix == 0).all()
            

def test_pawn_capture():
    board = Board()

    pawn_piece = board.add_piece(PAWN, WHITE, 6, 2)
    board.add_piece(ROOK, BLACK, 5, 1)

    move_matrix = get_moves(board, WHITE)

    true_moves = [Move(pawn_piece, coord) for coord in [[5, 1], [5, 2], [4, 2]]]

    move_matrix_contains_exactly(move_matrix, true_moves)

    board.add_piece(PAWN, BLACK, 5, 3)
    move_matrix = get_moves(board, WHITE)
    true_moves.append(Move(pawn_piece, [5, 3]))

    move_matrix_contains_exactly(move_matrix, true_moves)


def test_pawn_pins():
    board = Board()

    board.add_piece(KING, WHITE, 5, 5)
    pawn_piece = board.add_piece(PAWN, WHITE, 4, 4)

    board.add_piece(BISHOP, BLACK, 2, 2)
    move_matrix = get_moves(board, WHITE)

    piece_can_move_exactly(pawn_piece, move_matrix, [])

    board.add_piece(BISHOP, BLACK, 3, 3)
    move_matrix = get_moves(board, WHITE)

    piece_can_move_exactly(pawn_piece, move_matrix, [Move(pawn_piece, [3, 3])])

    pawn_piece2 = board.add_piece(PAWN, WHITE, 4, 5)
    board.add_piece(PAWN, BLACK, 3, 4)
    move_matrix = get_moves(board, WHITE)

    piece_can_move_exactly(pawn_piece2, move_matrix, [Move(pawn_piece2, coord) for coord in [[3, 4], [3, 5]]])

    board.add_piece(ROOK, BLACK, 0, 5)  # pins pawn
    move_matrix = get_moves(board, WHITE)

    piece_can_move_exactly(pawn_piece2, move_matrix, [Move(pawn_piece2, [3, 5])]) # can only move forward


def test_pawn_en_passant():
    board = Board()

    pawn_piece = board.add_piece(PAWN, WHITE, 3, 4)
    board.add_piece(PAWN, BLACK, 3, 3)

    move_matrix = get_moves(board, WHITE)

    piece_can_move_exactly(pawn_piece, move_matrix, [Move(pawn_piece, [2, 4])])

    # with en passant
    board.en_passant = np.array([3, 3])
    move_matrix = get_moves(board, WHITE)
    piece_can_move_exactly(pawn_piece, move_matrix, [Move(pawn_piece, [2, 4]), Move(pawn_piece, [2, 3])])

    # pin en passant piece
    board.add_piece(KING, WHITE, 5, 5)
    board.add_piece(BISHOP, BLACK, 0, 0)

    move_matrix = get_moves(board, WHITE)
    piece_can_move_exactly(pawn_piece, move_matrix, [Move(pawn_piece, [2, 4])])

    # vertical pin, all moves should still be available
    board.remove_piece_at(5, 5)
    board.remove_piece_at(0, 0)
    board.add_piece(KING, WHITE, 5, 3)
    board.add_piece(QUEEN, BLACK, 0, 3)

    move_matrix = get_moves(board, WHITE)
    piece_can_move_exactly(pawn_piece, move_matrix, [Move(pawn_piece, [2, 4]), Move(pawn_piece, [2, 3])])

    # rare lateral double pawn pin
    board.remove_piece_at(5, 3)
    board.remove_piece_at(0, 3)
    board.add_piece(KING, WHITE, 3, 0)
    board.add_piece(QUEEN, BLACK, 3, 6)

    move_matrix = get_moves(board, WHITE)
    piece_can_move_exactly(pawn_piece, move_matrix, [Move(pawn_piece, [2, 4])]) # only forward move

    # add blocking piece
    board.add_piece(ROOK, WHITE, 3, 2)

    move_matrix = get_moves(board, WHITE)
    piece_can_move_exactly(pawn_piece, move_matrix, [Move(pawn_piece, [2, 4]), Move(pawn_piece, [2, 3])])

    # test blocking piece on other side
    board.remove_piece_at(3, 2)
    board.add_piece(ROOK, WHITE, 3, 5)
    piece_can_move_exactly(pawn_piece, move_matrix, [Move(pawn_piece, [2, 4]), Move(pawn_piece, [2, 3])])


def test_knight_moves():
    board = Board()

    knight_piece = board.add_piece(KNIGHT, WHITE, 4, 4)

    true_moves = [Move(knight_piece, knight_piece.coord + [rank_diff, file_diff])
                    for rank_diff in [-2, -1, 1, 2]
                    for file_diff in [-2, -1, 1, 2]
                    if abs(rank_diff) != abs(file_diff)]
    
    move_matrix = get_moves(board, WHITE)
    piece_can_move_exactly(knight_piece, move_matrix, true_moves)

    # friendly blocking piece
    board.add_piece(PAWN, WHITE, 3, 2)
    true_moves.remove(Move(knight_piece, [3, 2]))
    move_matrix = get_moves(board, WHITE)
    piece_can_move_exactly(knight_piece, move_matrix, true_moves)

    # opponent piece can be captured
    board.add_piece(PAWN, BLACK, 5, 6)
    move_matrix = get_moves(board, WHITE)
    piece_can_move_exactly(knight_piece, move_matrix, true_moves)

    # pinned
    board.add_piece(KING, WHITE, 7, 7)
    board.add_piece(BISHOP, BLACK, 1, 1)
    move_matrix = get_moves(board, WHITE)
    piece_can_move_exactly(knight_piece, move_matrix, [])

    # knight on edge
    knight_piece = board.add_piece(KNIGHT, WHITE, 0, 0)
    true_moves = [Move(knight_piece, move) for move in [[2, 1], [1, 2]]]
    move_matrix = get_moves(board, WHITE)
    piece_can_move_exactly(knight_piece, move_matrix, true_moves)


def test_bishop_moves():
    board = Board()

    bishop_piece = board.add_piece(BISHOP, WHITE, 2, 3)

    true_moves = []

    for diff in np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]):
        to_coord = bishop_piece.coord + diff
        while within_bounds(*to_coord):
            true_moves.append(Move(bishop_piece, to_coord))
            to_coord = to_coord + diff
    
    move_matrix = get_moves(board, WHITE)

    piece_can_move_exactly(bishop_piece, move_matrix, true_moves)

    board.add_piece(KING, WHITE, 4, 5)
    true_moves.remove(Move(bishop_piece, [4, 5]))
    true_moves.remove(Move(bishop_piece, [5, 6]))
    true_moves.remove(Move(bishop_piece, [6, 7]))
    move_matrix = get_moves(board, WHITE)
    piece_can_move_exactly(bishop_piece, move_matrix, true_moves)

    board.add_piece(BISHOP, BLACK, 0, 1)
    for diff in np.array([[-1, 1], [1, -1]]):
        to_coord = bishop_piece.coord + diff
        while within_bounds(*to_coord):
            true_moves.remove(Move(bishop_piece, to_coord))
            to_coord = to_coord + diff
    
    move_matrix = get_moves(board, WHITE)
    piece_can_move_exactly(bishop_piece, move_matrix, true_moves)

    board.remove_piece_at(4, 5)
    board.add_piece(KING, WHITE, 2, 7)
    board.add_piece(ROOK, BLACK, 2, 0)
    move_matrix = get_moves(board, WHITE)
    piece_can_move_exactly(bishop_piece, move_matrix, [])


def test_rook_moves():
    board = Board()

    rook_piece = board.add_piece(ROOK, WHITE, 2, 3)

    true_moves = []

    for diff in np.array([[0, 1], [0, -1], [1, 0], [-1, 0]]):
        to_coord = rook_piece.coord + diff
        while within_bounds(*to_coord):
            true_moves.append(Move(rook_piece, to_coord))
            to_coord = to_coord + diff
    
    move_matrix = get_moves(board, WHITE)
    piece_can_move_exactly(rook_piece, move_matrix, true_moves)

    board.add_piece(KING, WHITE, 6, 3)
    true_moves.remove(Move(rook_piece, [6, 3]))
    true_moves.remove(Move(rook_piece, [7, 3]))

    move_matrix = get_moves(board, WHITE)
    piece_can_move_exactly(rook_piece, move_matrix, true_moves)

    board.add_piece(ROOK, BLACK, 0, 3)
    for diff in np.array([[0, 1], [0, -1]]):
        to_coord = rook_piece.coord + diff
        while within_bounds(*to_coord):
            true_moves.remove(Move(rook_piece, to_coord))
            to_coord = to_coord + diff
    
    move_matrix = get_moves(board, WHITE)
    piece_can_move_exactly(rook_piece, move_matrix, true_moves)

    board.remove_piece_at(6, 3)
    board.add_piece(KING, WHITE, 6, 7)
    board.add_piece(BISHOP, BLACK, 0, 1)

    move_matrix = get_moves(board, WHITE)
    piece_can_move_exactly(rook_piece, move_matrix, [])


def test_queen_moves():
    board = Board()

    queen_piece = board.add_piece(QUEEN, WHITE, 2, 3)

    true_moves = []

    for diff in np.array([[-1, -1], [-1, 1], [1, -1], [1, 1], [0, 1], [0, -1], [1, 0], [-1, 0]]):
        to_coord = queen_piece.coord + diff
        while within_bounds(*to_coord):
            true_moves.append(Move(queen_piece, to_coord))
            to_coord = to_coord + diff
    
    move_matrix = get_moves(board, WHITE)

    piece_can_move_exactly(queen_piece, move_matrix, true_moves)

    board.add_piece(KING, WHITE, 4, 5)
    true_moves.remove(Move(queen_piece, [4, 5]))
    true_moves.remove(Move(queen_piece, [5, 6]))
    true_moves.remove(Move(queen_piece, [6, 7]))
    move_matrix = get_moves(board, WHITE)
    piece_can_move_exactly(queen_piece, move_matrix, true_moves)

    board.add_piece(BISHOP, BLACK, 0, 1)
    for diff in np.array([[-1, 1], [1, -1], [0, 1], [0, -1], [1, 0], [-1, 0]]):
        to_coord = queen_piece.coord + diff
        while within_bounds(*to_coord):
            true_moves.remove(Move(queen_piece, to_coord))
            to_coord = to_coord + diff
    
    move_matrix = get_moves(board, WHITE)
    piece_can_move_exactly(queen_piece, move_matrix, true_moves)

    board.remove_piece_at(4, 5)
    board.add_piece(KING, WHITE, 2, 7)
    board.add_piece(ROOK, BLACK, 2, 0)

    true_moves = []
    for diff in np.array([[0, 1], [0, -1]]):
        to_coord = queen_piece.coord + diff
        while within_bounds(*to_coord):
            true_moves.append(Move(queen_piece, to_coord))
            to_coord = to_coord + diff

    true_moves.remove(Move(queen_piece, [2, 7]))

    move_matrix = get_moves(board, WHITE)
    piece_can_move_exactly(queen_piece, move_matrix, true_moves)