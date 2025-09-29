from game.model import Board, Move
from game.constants import *

import pytest
import numpy as np


def test_king_move():
    board = Board()
    king_piece = board.add_piece(KING, WHITE, 4, 4)

    prev_state = board.get_state(WHITE)
    undo = board.move_piece(Move(king_piece, [4, 3]))

    assert (king_piece.coord == [4, 3]).all(), "king in wrong position"
    assert (board.get_king(WHITE).coord == [4, 3]).all(), "king in wrong position"
    assert not (board.get_state(WHITE) == prev_state).all(), "board state not changed"

    undo()

    assert (king_piece.coord == [4, 4]).all(), "king in wrong position"
    assert (board.get_king(WHITE).coord == [4, 4]).all(), "king in wrong position"
    assert (board.get_state(WHITE) == prev_state).all(), "board not reset properly"


def test_king_take():
    board = Board()
    king_piece = board.add_piece(KING, WHITE, 4, 4)
    pawn_piece = board.add_piece(PAWN, BLACK, 4, 3)

    prev_state = board.get_state(WHITE)
    undo = board.move_piece(Move(king_piece, [4, 3]))

    assert len(board.of_team_and_type(BLACK, PAWN)) == 0, "pawn not removed"
    assert board.coord_map[4, 3] == king_piece, "king not in correct position"
    assert (king_piece.coord == [4, 3]).all(), "king in wrong position"
    assert (board.get_king(WHITE).coord == [4, 3]).all(), "king in wrong position"
    assert not (board.get_state(WHITE) == prev_state).all(), "board state not changed"

    undo()

    assert len(board.of_team_and_type(BLACK, PAWN)) == 1, "pawn not re-added"
    assert board.coord_map[4, 3] == pawn_piece, "pawn not re-added to correct position"
    assert (king_piece.coord == [4, 4]).all(), "king in wrong position"
    assert (board.get_king(WHITE).coord == [4, 4]).all(), "king in wrong position"
    assert (board.get_state(WHITE) == prev_state).all(), "board not reset properly"


@pytest.mark.parametrize("team,rank", [(WHITE, 7), (BLACK, 0)])
def test_castle(team, rank):
    board = Board()
    king_piece = board.add_piece(KING, team, rank, 4)
    rook1 = board.add_piece(ROOK, team, rank, 7)
    rook2 = board.add_piece(ROOK, team, rank, 0)

    board.king_side_castle[team] = True
    board.queen_side_castle[team] = True

    prev_state = board.get_state(team)

    undo = board.move_piece(Move(king_piece, [rank, 6]))

    assert (rook1.coord == [rank, 5]).all(), f"rook1 not moved after castle, {rook1.coord}"
    assert (king_piece.coord == [rank, 6]).all(), "king not moved after castle"
    assert board.coord_map[rank, 5] == rook1, "rook not in coord map"
    assert board.coord_map[rank, 6] == king_piece, "king not in coord map"
    assert not (board.get_state(team) == prev_state).all(), "state not changed"

    undo()

    assert (rook1.coord == [rank, 7]).all(), "rook1 not moved after undo castle"
    assert (king_piece.coord == [rank, 4]).all(), "king not moved after undo castle"
    assert board.coord_map[rank, 7] == rook1, "rook not in coord map"
    assert board.coord_map[rank, 4] == king_piece, "king not in coord map"
    assert (board.get_state(team) == prev_state).all(), "state not changed back"

    undo = board.move_piece(Move(king_piece, [rank, 2]))

    assert (rook2.coord == [rank, 3]).all(), f"rook2 not moved after castle, {rook2.coord}"
    assert (king_piece.coord == [rank, 2]).all(), "king not moved after castle"
    assert board.coord_map[rank, 3] == rook2, "rook not in coord map"
    assert board.coord_map[rank, 2] == king_piece, "king not in coord map"
    assert not (board.get_state(team) == prev_state).all(), "state not changed"

    undo()

    assert (rook2.coord == [rank, 0]).all(), "rook1 not moved after undo castle"
    assert (king_piece.coord == [rank, 4]).all(), "king not moved after undo castle"
    assert board.coord_map[rank, 0] == rook2, "rook not in coord map"
    assert board.coord_map[rank, 4] == king_piece, "king not in coord map"
    assert (board.get_state(team) == prev_state).all(), "state not changed back"


@pytest.mark.parametrize("team,start_rank,end_rank,pawn_dir,rank,file", 
                         [(team, start_rank, end_rank, pawn_dir, rank, file) 
                          for team, start_rank, end_rank, pawn_dir in [(WHITE, 6, 1, -1), (BLACK, 1, 6, 1)] 
                          for rank in range(1, 7) for file in range(8)])
def test_pawn_move(team, start_rank, end_rank, pawn_dir, rank, file):
    board = Board()
    pawn_piece = board.add_piece(PAWN, team, rank, file)

    if rank == end_rank:
        with pytest.raises(Exception):
            board.move_piece(Move(pawn_piece, [rank + pawn_dir, file]))
    else:
        prev_state = board.get_state(team)

        undo = board.move_piece(Move(pawn_piece, [rank + pawn_dir, file]))

        assert board.coord_map[rank + pawn_dir, file] == pawn_piece
        assert board.coord_map.get((rank, file)) is None
        assert (pawn_piece.coord == [rank + pawn_dir, file]).all()
        assert not (board.get_state(team) == prev_state).all()

        undo()

        assert board.coord_map[rank, file] == pawn_piece
        assert board.coord_map.get((rank + pawn_dir, file)) is None
        assert (pawn_piece.coord == [rank, file]).all()
        assert (board.get_state(team) == prev_state).all()

        if rank == start_rank:
            undo = board.move_piece(Move(pawn_piece, [rank + 2 * pawn_dir, file]))

            assert board.coord_map[rank + 2 * pawn_dir, file] == pawn_piece
            assert board.coord_map.get((rank, file)) is None
            assert (pawn_piece.coord == [rank + 2 * pawn_dir, file]).all()
            assert not (board.get_state(team) == prev_state).all()

            undo()

            assert board.coord_map[rank, file] == pawn_piece
            assert board.coord_map.get((rank + 2 * pawn_dir, file)) is None
            assert (pawn_piece.coord == [rank, file]).all()
            assert (board.get_state(team) == prev_state).all()


@pytest.mark.parametrize("team,pawn_dir", [(WHITE, -1), (BLACK, 1)])
def test_pawn_take(team, pawn_dir):
    board = Board()
    pawn = board.add_piece(PAWN, team, 4, 4)
    enemy_pawn1 = board.add_piece(PAWN, other_team(team), 4 + pawn_dir, 3)
    enemy_pawn2 = board.add_piece(PAWN, other_team(team), 4 + pawn_dir, 5)

    prev_state = board.get_state(team)

    undo = board.move_piece(Move(pawn, enemy_pawn1.coord))

    assert board.coord_map[*enemy_pawn1.coord] == pawn
    assert (pawn.coord == enemy_pawn1.coord).all()
    assert board.coord_map.get((4, 4)) is None
    assert not (board.get_state(team) == prev_state).all()

    undo()

    assert board.coord_map[*enemy_pawn1.coord] == enemy_pawn1
    assert (pawn.coord == [4, 4]).all()
    assert board.coord_map.get((4, 4)) == pawn
    assert (board.get_state(team) == prev_state).all()

    undo = board.move_piece(Move(pawn, enemy_pawn2.coord))

    assert board.coord_map[*enemy_pawn2.coord] == pawn
    assert (pawn.coord == enemy_pawn2.coord).all()
    assert board.coord_map.get((4, 4)) is None
    assert not (board.get_state(team) == prev_state).all()

    undo()

    assert board.coord_map[*enemy_pawn2.coord] == enemy_pawn2
    assert (pawn.coord == [4, 4]).all()
    assert board.coord_map.get((4, 4)) == pawn
    assert (board.get_state(team) == prev_state).all()


@pytest.mark.parametrize("team,pawn_dir,rank", [(BLACK, 1, 5), (WHITE, -1, 4)])
def test_en_passant(team, pawn_dir, rank):
    board = Board()
    pawn = board.add_piece(PAWN, team, rank, 4)
    enemy_pawn1 = board.add_piece(PAWN, other_team(team), rank, 3)
    enemy_pawn2 = board.add_piece(PAWN, other_team(team), rank, 5)

    board.en_passant = np.array([rank, 3])

    prev_state = board.get_state(team)

    undo = board.move_piece(Move(pawn, enemy_pawn1.coord + [pawn_dir, 0]))

    assert board.coord_map[*(enemy_pawn1.coord + [pawn_dir, 0])] == pawn
    assert (pawn.coord == (enemy_pawn1.coord + [pawn_dir, 0])).all()
    assert board.coord_map.get((rank, 4)) is None
    assert not (board.get_state(team) == prev_state).all()

    undo()

    assert board.coord_map[*enemy_pawn1.coord] == enemy_pawn1
    assert (pawn.coord == [rank, 4]).all()
    assert board.coord_map.get((rank, 4)) == pawn
    assert (board.get_state(team) == prev_state).all()

    board.en_passant = np.array([rank, 5])
    undo = board.move_piece(Move(pawn, enemy_pawn2.coord + [pawn_dir, 0]))

    assert board.coord_map[*(enemy_pawn2.coord + [pawn_dir, 0])] == pawn
    assert (pawn.coord == (enemy_pawn2.coord + [pawn_dir, 0])).all()
    assert board.coord_map.get((rank, 4)) is None
    assert not (board.get_state(team) == prev_state).all()

    undo()

    assert board.coord_map[*enemy_pawn2.coord] == enemy_pawn2
    assert (pawn.coord == [rank, 4]).all()
    assert board.coord_map.get((rank, 4)) == pawn
    assert (board.get_state(team) == prev_state).all()


@pytest.mark.parametrize("team,pawn_dir,start_rank,promote", [(team, pawn_dir, start_rank, promote) for team, pawn_dir, start_rank in [(WHITE, -1, 1), (BLACK, 1, 6)] for promote in range(4)])
def test_promote(team, pawn_dir, start_rank, promote):
    promote_piece_type = [QUEEN, ROOK, BISHOP, KNIGHT][promote]

    board = Board()
    pawn = board.add_piece(PAWN, team, start_rank, 4)
    other_queen = board.add_piece(QUEEN, other_team(team), start_rank + pawn_dir, 5)

    prev_state = board.get_state(team)

    undo = board.move_piece(Move(pawn, pawn.coord + [pawn_dir, 0], promote))

    assert board.coord_map.get((start_rank, 4)) is None
    assert board.coord_map[start_rank + pawn_dir, 4] == pawn
    assert pawn.piece_type == promote_piece_type
    assert board.coord_map[start_rank + pawn_dir, 4].piece_type == promote_piece_type
    assert not (board.get_state(team) == prev_state).all()

    undo()

    assert board.coord_map[start_rank, 4] == pawn
    assert board.coord_map.get((start_rank + pawn_dir, 4)) is None
    assert pawn.piece_type == PAWN
    assert board.coord_map[start_rank, 4].piece_type == PAWN
    assert (board.get_state(team) == prev_state).all()
    
    undo = board.move_piece(Move(pawn, other_queen.coord, promote))

    assert board.coord_map.get((start_rank, 4)) is None
    assert board.coord_map[*other_queen.coord] == pawn
    assert pawn.piece_type == promote_piece_type
    assert board.coord_map[*other_queen.coord].piece_type == promote_piece_type
    assert not (board.get_state(team) == prev_state).all()

    undo()

    assert board.coord_map[start_rank, 4] == pawn
    assert board.coord_map[*other_queen.coord] == other_queen
    assert pawn.piece_type == PAWN
    assert board.coord_map[start_rank, 4].piece_type == PAWN
    assert (board.get_state(team) == prev_state).all()