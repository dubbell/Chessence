import unittest

import numpy as np

from game.model import Board, Move

from game.constants import *
from game.utils import within_bounds

from game.move_calc import get_moves

from test.utils import CustomAssert



class TestMoveCalc(unittest.TestCase, CustomAssert):

    def test_king_moves(self):
        for king_coord in np.array([[rank, file] for rank in range(8) for file in range(8)]):
            board = Board()
            king_piece = board.add_piece(KING, WHITE, *king_coord)

            moves = get_moves(board, WHITE)

            true_moves = [Move(move) for move in king_coord + [
                    [rank_diff, file_diff]
                    for rank_diff in [-1, 0, 1]
                    for file_diff in [-1, 0, 1]
                    if rank_diff != 0 or file_diff != 0]
                if within_bounds(*move)]

            self.assertEqual(len(moves[king_piece]), len(true_moves))

            self.assertContainsExactly(true_moves, moves[king_piece])


    def test_king_moves_controlled(self):
        board = Board()
        king_coord = np.array([4, 4])
        king_piece = board.add_piece(KING, WHITE, *king_coord)

        board.add_piece(ROOK, BLACK, 3, 0)

        moves = get_moves(board, WHITE)

        true_moves = [Move(move) for move in king_coord + [
                    [rank_diff, file_diff]
                    for rank_diff in [-1, 0, 1]
                    for file_diff in [-1, 0, 1]
                    if (rank_diff != 0 or file_diff != 0) and rank_diff != -1]]
        
        self.assertEqual(len(moves[king_piece]), 5)
        self.assertContainsExactly(true_moves, moves[king_piece])

        board.add_piece(BISHOP, BLACK, 2, 3)

        moves = get_moves(board, WHITE)

        true_moves.remove(Move([4, 5]))

        self.assertEqual(len(moves[king_piece]), 4)
        self.assertContainsExactly(true_moves, moves[king_piece])
    
        # pawn can move
        pawn_piece = board.add_piece(PAWN, WHITE, 4, 7)
        
        moves = get_moves(board, WHITE)

        self.assertEqual(len(moves), 2)  # both king and pawn can move
        self.assertContainsExactly(true_moves, moves[king_piece])
        self.assertContainsExactly([Move([3, 7])], moves[pawn_piece])

        # queen checks king, pawn can no longer move
        board.add_piece(QUEEN, BLACK, 2, 2)

        moves = get_moves(board, WHITE)

        true_moves.remove(Move([5, 5]))

        self.assertEqual(len(moves), 1)  # only king can move now
        self.assertEqual(len(moves[king_piece]), 3)
        self.assertContainsExactly(true_moves, moves[king_piece])

        board.add_piece(KNIGHT, BLACK, 6, 2)

        moves = get_moves(board, WHITE)

        true_moves.remove(Move([5, 4]))
        true_moves.remove(Move([4, 3]))

        self.assertEqual(len(moves[king_piece]), 1)
        self.assertContainsExactly(true_moves, moves[king_piece])

        board.add_piece(KING, BLACK, 6, 4)

        moves = get_moves(board, WHITE)

        self.assertIsNone(moves)  # checkmate

        board.remove_piece_at(2, 2)  # remove checking piece
        board.remove_piece_at(4, 7)  # remove white pawn

        moves = get_moves(board, WHITE)

        self.assertEqual(len(moves), 0)  # stalemate

    
    def test_king_moves_blocked(self):
        board = Board()
        king_coord = np.array([4, 4])
        king_piece = board.add_piece(KING, WHITE, *king_coord)

        board.add_piece(PAWN, WHITE, 4, 3)

        moves = get_moves(board, WHITE)

        true_moves = [Move(move) for move in king_coord + [
                [rank_diff, file_diff]
                for rank_diff in [-1, 0, 1]
                for file_diff in [-1, 0, 1]
                if (rank_diff != 0 or file_diff != 0)]]
        
        true_moves.remove(Move([4, 3]))

        self.assertEqual(len(moves), 2)
        self.assertEqual(len(moves[king_piece]), len(true_moves))
        self.assertContainsExactly(moves[king_piece], true_moves)

        board.add_piece(PAWN, WHITE, 5, 5)

        moves = get_moves(board, WHITE)

        true_moves.remove(Move([5, 5]))

        self.assertEqual(len(moves), 3)
        self.assertEqual(len(moves[king_piece]), len(true_moves))
        self.assertContainsExactly(moves[king_piece], true_moves)

        board.add_piece(BISHOP, BLACK, 5, 4)

        moves = get_moves(board, WHITE)

        true_moves.remove(Move([4, 5]))  # square controlled by bishop

        self.assertEqual(len(moves), 3)
        self.assertEqual(len(moves[king_piece]), len(true_moves))
        self.assertContainsExactly(moves[king_piece], true_moves)


    def test_pawn_moves(self):
        for team, ranks in zip([WHITE, BLACK], [[6, 5, 4], [1, 2, 3]]):
            for file in range(8):
                board = Board() 

                pawn_piece = board.add_piece(PAWN, team, ranks[0], file)
                moves = get_moves(board, team)
                true_moves = [Move([rank, file]) for rank in ranks[1:]]
                self.assertContainsExactly(moves[pawn_piece], true_moves)

                board.add_piece(PAWN, other_team(team), ranks[2], file)
                moves = get_moves(board, team)
                self.assertContainsExactly(moves[pawn_piece], true_moves[:1])

                board.add_piece(PAWN, other_team(team), ranks[1], file)
                moves = get_moves(board, team)
                self.assertEqual(len(moves), 0)
                

    def test_pawn_capture(self):
        board = Board()

        pawn_piece = board.add_piece(PAWN, WHITE, 6, 2)
        board.add_piece(ROOK, BLACK, 5, 1)

        moves = get_moves(board, WHITE)

        true_moves = [Move(coord) for coord in [[5, 1], [5, 2], [4, 2]]]

        self.assertContainsExactly(moves[pawn_piece], true_moves)

        board.add_piece(PAWN, BLACK, 5, 3)
        moves = get_moves(board, WHITE)
        true_moves.append(Move([5, 3]))

        self.assertContainsExactly(moves[pawn_piece], true_moves)
    

    def test_pawn_pins(self):
        board = Board()

        board.add_piece(KING, WHITE, 5, 5)
        pawn_piece = board.add_piece(PAWN, WHITE, 4, 4)

        board.add_piece(BISHOP, BLACK, 2, 2)
        moves = get_moves(board, WHITE)

        self.assertTrue(pawn_piece not in moves)

        board.add_piece(BISHOP, BLACK, 3, 3)
        moves = get_moves(board, WHITE)

        self.assertContainsExactly(moves[pawn_piece], [Move([3, 3])])

        pawn_piece2 = board.add_piece(PAWN, WHITE, 4, 5)
        board.add_piece(PAWN, BLACK, 3, 4)
        moves = get_moves(board, WHITE)

        self.assertContainsExactly(moves[pawn_piece2], [Move(coord) for coord in [[3, 4], [3, 5]]])

        board.add_piece(ROOK, BLACK, 0, 5)  # pins pawn
        moves = get_moves(board, WHITE)

        self.assertContainsExactly(moves[pawn_piece2], [Move([3, 5])])
    

    def test_pawn_en_passant(self):
        board = Board()

        pawn_piece = board.add_piece(PAWN, WHITE, 3, 4)
        board.add_piece(PAWN, BLACK, 3, 3)

        moves = get_moves(board, WHITE)

        self.assertContainsExactly(moves[pawn_piece], [Move([2, 4])])

        # with en passant
        moves = get_moves(board, WHITE, [3, 3])
        self.assertContainsExactly(moves[pawn_piece], [Move([2, 4]), Move([2, 3])])

        # pin en passant piece
        board.add_piece(KING, WHITE, 5, 5)
        board.add_piece(BISHOP, BLACK, 0, 0)

        moves = get_moves(board, WHITE, [3, 3])
        self.assertContainsExactly(moves[pawn_piece], [Move([2, 4])])

        # vertical pin, all moves should still be available
        board.remove_piece_at(5, 5)
        board.remove_piece_at(0, 0)
        board.add_piece(KING, WHITE, 5, 3)
        board.add_piece(QUEEN, BLACK, 0, 3)

        moves = get_moves(board, WHITE, [3, 3])
        self.assertContainsExactly(moves[pawn_piece], [Move([2, 4]), Move([2, 3])])

        # rare lateral double pawn pin
        board.remove_piece_at(5, 3)
        board.remove_piece_at(0, 3)
        board.add_piece(KING, WHITE, 3, 0)
        board.add_piece(QUEEN, BLACK, 3, 6)

        moves = get_moves(board, WHITE, [3, 3])
        self.assertContainsExactly(moves[pawn_piece], [Move([2, 4])]) # only forward move

        # add blocking piece
        board.add_piece(ROOK, WHITE, 3, 2)

        moves = get_moves(board, WHITE, [3, 3])
        self.assertContainsExactly(moves[pawn_piece], [Move([2, 4]), Move([2, 3])])

        # test blocking piece on other side
        board.remove_piece_at(3, 2)
        board.add_piece(ROOK, WHITE, 3, 5)
        self.assertContainsExactly(moves[pawn_piece], [Move([2, 4]), Move([2, 3])])
