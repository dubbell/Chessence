import unittest
from model import Board
from constants import *

from king_state import king_state


class KingStateTest(unittest.TestCase):

    def test_control(self):
        board = Board()
        board.add_piece(KING, WHITE, 4, 4)
        board.add_piece(BISHOP, BLACK, 1, 1)
        board.add_piece(BISHOP, BLACK, 2, 1)

        controlled, pins = king_state(board, WHITE)

        self.assertTrue((controlled == [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 1]]).all())


unittest.main()
