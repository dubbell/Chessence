import unittest

import numpy as np
from numpy.testing import assert_array_equal

from game.model import Board, Move
from game.constants import *
from game.utils import within_bounds

from game.move import get_moves

from test.utils import contains_exactly



class TestMoves(unittest.TestCase):

    def test_king_moves(self):
        for king_coord in np.array([[rank, file] for rank in range(8) for file in range(8)]):
            board = Board()
            board.add_piece(KING, WHITE, *king_coord)

            moves = get_moves(board, WHITE)

            true_moves = [Move(0, move) for move in king_coord + [
                    [rank_diff, file_diff]
                    for rank_diff in [-1, 0, 1]
                    for file_diff in [-1, 0, 1]
                    if rank_diff != 0 or file_diff != 0]
                if within_bounds(*move)]

            self.assertEqual(len(moves), len(true_moves))

            for move in moves:
                self.assertIn(move, true_moves)


    def test_king_moves_controlled(self):
        board = Board()
        king_coord = np.array([4, 4])
        board.add_piece(KING, WHITE, *king_coord)

        board.add_piece(ROOK, BLACK, 3, 0)

        moves = get_moves(board, WHITE)

        true_moves = [Move(0, move) for move in king_coord + [
                    [rank_diff, file_diff]
                    for rank_diff in [-1, 0, 1]
                    for file_diff in [-1, 0, 1]
                    if (rank_diff != 0 or file_diff != 0) and rank_diff != -1]]
        
        self.assertEqual(len(moves), 5)
        for move in moves:
            self.assertIn(move, true_moves)

        board.add_piece(BISHOP, BLACK, 2, 3)

        moves = get_moves(board, WHITE)

        true_moves.remove(Move(0, [4, 5]))

        self.assertEqual(len(moves), 4)
        for move in moves:
            self.assertIn(move, true_moves)
    
        # to make sure only king moves are available when king is checked
        board.add_piece(PAWN, WHITE, 4, 7)
        
        moves = get_moves(board, WHITE)

        self.assertEqual(len(moves), 5)
        self.assertIn(Move(1, [3, 7]), moves)  # pawn move
        moves.remove(Move(1, [3, 7]))
        for move in moves:
            self.assertIn(move, true_moves)

        # queen checks king
        board.add_piece(QUEEN, BLACK, 2, 2)

        moves = get_moves(board, WHITE)

        true_moves.remove(Move(0, [5, 5]))

        self.assertEqual(len(moves), 3)
        for move in moves:
            self.assertIn(move, true_moves)
        self.assertNotIn(Move(1, [3, 7]), moves)  # pawn move should no longer be available
        
        board.add_piece(KNIGHT, BLACK, 6, 2)

        moves = get_moves(board, WHITE)

        true_moves.remove(Move(0, [5, 4]))
        true_moves.remove(Move(0, [4, 3]))

        self.assertEqual(len(moves), 1)
        for move in moves:
            self.assertIn(move, true_moves)
        
        board.add_piece(KING, BLACK, 6, 4)

        moves = get_moves(board, WHITE)

        self.assertIsNone(moves)  # checkmate

        board.remove_piece_at(BLACK, 2, 2)  # remove checking piece
        board.remove_piece_at(WHITE, 4, 7)  # remove white pawn

        moves = get_moves(board, WHITE)

        self.assertEqual(len(moves), 0)  # stalemate

    
    def test_king_moves_blocked(self):
        board = Board()
        king_coord = np.array([4, 4])
        board.add_piece(KING, WHITE, *king_coord)

        board.add_piece(PAWN, WHITE, 4, 3)

        moves = get_moves(board, WHITE)

        true_moves = [Move(0, move) for move in king_coord + [
                [rank_diff, file_diff]
                for rank_diff in [-1, 0, 1]
                for file_diff in [-1, 0, 1]
                if (rank_diff != 0 or file_diff != 0)]]
        
        true_moves.remove(Move(0, [4, 3]))

        self.assertEqual(len(moves), len(true_moves)+1)
        for move in moves:
            if move.piece_index == 0:
                self.assertIn(move, true_moves)

        board.add_piece(PAWN, WHITE, 5, 5)

        moves = get_moves(board, WHITE)

        true_moves.remove(Move(0, [5, 5]))

        self.assertEqual(len(moves), len(true_moves)+2)
        for move in moves:
            if move.piece_index == 0:
                self.assertIn(move, true_moves)

        board.add_piece(BISHOP, BLACK, 5, 4)

        moves = get_moves(board, WHITE)

        true_moves.remove(Move(0, [4, 5]))

        self.assertEqual(len(moves), len(true_moves)+2)
        for move in moves:
            if move.piece_index == 0:
                self.assertIn(move, true_moves)


    def test_pawn_moves(self):
        for team, ranks in zip([WHITE, BLACK], [[6, 5, 4], [1, 2, 3]]):
            for file in range(8):
                board = Board() 

                board.add_piece(PAWN, team, ranks[0], file)
                moves = get_moves(board, team)
                true_moves = [Move(0, [rank, file]) for rank in ranks[1:]]
                self.assertTrue(contains_exactly(moves, true_moves))

                board.add_piece(PAWN, int(not team), ranks[2], file)
                moves = get_moves(board, team)
                self.assertTrue(contains_exactly(moves, true_moves[:1]))

                board.add_piece(PAWN, int(not team), ranks[1], file)
                moves = get_moves(board, team)
                self.assertEqual(len(moves), 0)
                

    def test_pawn_capture(self):
        board = Board()

        board.add_piece(PAWN, WHITE, 6, 2)
        board.add_piece(ROOK, BLACK, 5, 1)

        moves = get_moves(board, WHITE)

        true_moves = [Move(0, coord) for coord in [[5, 1], [5, 2], [4, 2]]]

        self.assertTrue(contains_exactly(moves, true_moves))

        board.add_piece(PAWN, BLACK, 5, 3)
        moves = get_moves(board, WHITE)
        true_moves.append(Move(0, [5, 3]))

        self.assertTrue(contains_exactly(moves, true_moves))
    

    def test_pawn_pins(self):
        board = Board()

        board.add_piece(KING, WHITE, 5, 5)
        board.add_piece(PAWN, WHITE, 4, 4)

        board.add_piece(BISHOP, BLACK, 3, 3)

        pawn_moves = [move for move in get_moves(board, WHITE) if move.piece_type == PAWN]

        true_moves = [Move()]

        self.assertTrue()
