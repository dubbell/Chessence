import unittest
import numpy as np
from numpy.testing import assert_array_equal
from model import Board
from constants import *

from king_state import king_state



class KingStateTest(unittest.TestCase):

    def test_diagonal_pins(self):
        king_coord = np.array([4, 4])
        for i, dir in enumerate(np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])):
            for pin_dist in range(1, 3):
                board = Board()
                board.add_piece(KING, WHITE, *king_coord)
                board.add_piece(BISHOP, BLACK, *(king_coord + dir * 3))
                board.add_piece(PAWN, WHITE, *(king_coord + dir * pin_dist))

                controlled, pin_coords, pin_dirs = king_state(board, WHITE)

                true_controlled = np.rot90(np.array([
                    [2 - pin_dist, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]]), -i)

                assert_array_equal(controlled, true_controlled)

                assert_array_equal(pin_coords, [king_coord + dir * pin_dist])
                assert_array_equal(pin_dirs, [dir])
    

    def test_lateral_pins(self):
        king_coord = np.array([4, 4])
        for i, dir in enumerate(np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])):
            for pin_dist in range(1, 3):
                board = Board()
                board.add_piece(KING, WHITE, *king_coord)
                board.add_piece(ROOK, BLACK, *(king_coord + dir * 3))
                board.add_piece(PAWN, WHITE, *(king_coord + dir * pin_dist))

                controlled, pin_coords, pin_dirs = king_state(board, WHITE)

                true_controlled = np.rot90(np.array([
                    [0, 2 - pin_dist, 0],
                    [0, 0, 0],
                    [0, 0, 0]]), -i)

                assert_array_equal(controlled, true_controlled)

                assert_array_equal(pin_coords, [king_coord + dir * pin_dist])
                assert_array_equal(pin_dirs, [dir])


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
        
    
    def test_pawn_control(self):
        for team in [WHITE, BLACK]:
            king_coord = [4, 4]
            for pawn_rank in np.arange(2, 4):
                for pawn_file in range(2, 7):
                    board = Board()
                    board.add_piece(KING, team, *king_coord)
                    board.add_piece(PAWN, int(not team), pawn_rank if team == WHITE else 8 - pawn_rank, pawn_file)

                    controlled, _, _ = king_state(board, team)

                    true_controlled = np.zeros((3, 3))
                    if pawn_file - 4 >= 0:
                        true_controlled[pawn_rank - 2, pawn_file - 4] = 1
                    if pawn_file - 2 <= 2:
                        true_controlled[pawn_rank - 2, pawn_file - 2] = 1
                    
                    if team == BLACK:
                        true_controlled = np.flip(true_controlled, axis=0)
                    
                    assert_array_equal(controlled, true_controlled)
                    

    def test_contact_knight(self):
        board = Board()
        board.add_piece(KING, WHITE, 4, 4)
        board.add_piece(KNIGHT, BLACK, 3, 3)

        controlled, _, _ = king_state(board, WHITE)

        true_controlled = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0]])
        
        assert_array_equal(controlled, true_controlled)

        board.add_piece(KNIGHT, BLACK, 5, 5)

        controlled, _, _ = king_state(board, WHITE)

        true_controlled = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]])
        
        assert_array_equal(controlled, true_controlled)

        board.add_piece(KNIGHT, BLACK, 5, 4)

        controlled, _, _ = king_state(board, WHITE)

        true_controlled = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [0, 1, 0]])
        
        assert_array_equal(controlled, true_controlled)

        board.add_piece(KNIGHT, BLACK, 4, 3)

        controlled, _, _ = king_state(board, WHITE)

        true_controlled = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [0, 1, 1]])
        
        assert_array_equal(controlled, true_controlled)


    def test_contact_bishop(self):
        board = Board()
        board.add_piece(KING, WHITE, 4, 4)
        board.add_piece(BISHOP, BLACK, 3, 3)

        controlled, _, _ = king_state(board, WHITE)

        true_controlled = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 1]])
        
        assert_array_equal(controlled, true_controlled)

        board.add_piece(BISHOP, BLACK, 3, 4)

        controlled, _, _ = king_state(board, WHITE)

        true_controlled = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 1]])
        
        assert_array_equal(controlled, true_controlled)

        board.add_piece(BISHOP, BLACK, 5, 3)

        controlled, _, _ = king_state(board, WHITE)

        true_controlled = np.array([
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 1]])
        
        assert_array_equal(controlled, true_controlled)


    def test_contact_rook(self):
        board = Board()
        board.add_piece(KING, WHITE, 4, 4)
        board.add_piece(ROOK, BLACK, 3, 3)

        controlled, _, _ = king_state(board, WHITE)

        true_controlled = np.array([
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 0]])
        
        assert_array_equal(controlled, true_controlled)
        
        board.add_piece(PAWN, WHITE, 3, 4)

        controlled, _, _ = king_state(board, WHITE)

        true_controlled = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0]])
        
        assert_array_equal(controlled, true_controlled)

        board.add_piece(ROOK, BLACK, 3, 5)

        controlled, _, _ = king_state(board, WHITE)

        true_controlled = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [1, 0, 1]])
        
        assert_array_equal(controlled, true_controlled)
        
        board.add_piece(ROOK, BLACK, 5, 5)

        controlled, _, _ = king_state(board, WHITE)

        true_controlled = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1]])
        
        assert_array_equal(controlled, true_controlled)


    def test_contact_queen(self):
        board = Board()
        board.add_piece(KING, WHITE, 4, 4)
        board.add_piece(QUEEN, BLACK, 3, 3)

        controlled, _, _ = king_state(board, WHITE)

        true_controlled = np.array([
            [0, 1, 1],
            [1, 1, 0],
            [1, 0, 1]])
        
        assert_array_equal(controlled, true_controlled)
        
        board.add_piece(PAWN, WHITE, 4, 3)

        controlled, _, _ = king_state(board, WHITE)

        true_controlled = np.array([
            [0, 1, 1],
            [1, 1, 0],
            [0, 0, 1]])
        
        assert_array_equal(controlled, true_controlled)
        
        board.add_piece(QUEEN, BLACK, 5, 4)

        controlled, _, _ = king_state(board, WHITE)

        true_controlled = np.array([
            [0, 1, 1],
            [1, 1, 1],
            [1, 0, 1]])
        
        assert_array_equal(controlled, true_controlled)


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(unittest.TestLoader().loadTestsFromTestCase(KingStateTest))
    # runner.run(KingStateTest('test_lateral_pins'))
