import unittest
from numpy.testing import assert_array_equal
from model import Board
from constants import *

from king_state import king_state



class KingStateTest(unittest.TestCase):

    def test_double_bishop(self):
        board = Board()
        board.add_piece(KING, WHITE, 4, 4)
        board.add_piece(BISHOP, BLACK, 1, 1)
        board.add_piece(BISHOP, BLACK, 2, 1)

        controlled, pin_coords, pin_dirs = king_state(board, WHITE)

        assert_array_equal(controlled, [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 1]])
        
        assert_array_equal(pin_coords, [])
        assert_array_equal(pin_dirs, [])


    def test_double_bishop_pin(self):
        board = Board()
        board.add_piece(KING, WHITE, 4, 4)
        board.add_piece(PAWN, WHITE, 3, 3)
        board.add_piece(BISHOP, BLACK, 1, 1)
        board.add_piece(BISHOP, BLACK, 2, 1)

        controlled, pin_coords, pin_dirs = king_state(board, WHITE)

        assert_array_equal(controlled, [
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0]])
        
        assert_array_equal(pin_coords, [[3, 3]])
        assert_array_equal(pin_dirs, [[-1, -1]])

    def test_bishop_rook(self):
        board = Board()
        board.add_piece(KING, BLACK, 4, 4)
        board.add_piece(BISHOP, WHITE, 1, 1)
        board.add_piece(ROOK, WHITE, 1, 5)

        controlled, pin_coords, pin_dirs = king_state(board, BLACK)

        assert_array_equal(controlled, [
            [1, 0, 1],
            [0, 1, 1],
            [0, 0, 1]])
        
        assert_array_equal(pin_coords, [])
        assert_array_equal(pin_dirs, [])

        
    def test_bishop_rook_block(self):
        board = Board()
        board.add_piece(KING, BLACK, 4, 4)
        board.add_piece(PAWN, BLACK, 3, 5)
        board.add_piece(BISHOP, WHITE, 1, 1)
        board.add_piece(ROOK, WHITE, 1, 5)

        controlled, pin_coords, pin_dirs = king_state(board, BLACK)

        assert_array_equal(controlled, [
            [1, 0, 1],
            [0, 1, 0],
            [0, 0, 1]])
        
        assert_array_equal(pin_coords, [])
        assert_array_equal(pin_dirs, [])
        

unittest.main()
