from ..utils.buffer import get_buffer_loader
from game.move_calc import get_moves
from game.model import Board, Move
from game.constants import *
from game.utils import other_team
from ..model.sac import SAC

import numpy as np
import mlflow


def get_move_matrix(move_map):
    """Simplified representation of all available moves. 
       Row = coordinate of piece that is being moved (select). 
       Col = coordinate that the selected piece can be moved to (target)."""
    
    move_matrix = np.zeros((64, 64), dtype=bool)
    for piece, moves in move_map.items():
        available_select = np.ravel_multi_index(piece.coord, (8, 8))
        for move in moves:
            available_target = np.ravel_multi_index(move.to_coord, (8, 8))
            move_matrix[available_select, available_target] = True
    
    return move_matrix


class MoveResult(Enum):
    CONTINUE = 0
    LOSS = 1
    DRAW = 2

    def __eq__(self, other):
        return isinstance(other, Enum) and self.value == other.value
    
    def __hash__(self):
        return self.value

CONTINUE = MoveResult.CONTINUE
LOSS = MoveResult.LOSS
DRAW = MoveResult.DRAW

def take_action(board : Board, agent : SAC, team : Team, en_passant : np.array):
    """
    Let agent take action on board for the given team.
    Input: the board, the agent, and the agent's team
    Output: en_passant
    """

    current_state = board.get_state()
    move_map = get_moves(board, team, en_passant)
    # no moves available because checkmate
    if move_map is None:
        return current_state, None, None, LOSS
    # no moves available because draw
    elif not move_map:
        return current_state, None, None, DRAW

    move_matrix = get_move_matrix(move_map)
    action = agent.sample_action(current_state, move_matrix, team)

    select = np.unravel_index(action[0], (8, 8))
    target = np.unravel_index(action[1], (8, 8))
   
    selected_piece = board.coord_map[*select]
    en_passant = board.move_piece(selected_piece, Move(*target))
    
    next_state = board.get_state()

    return next_state, action, en_passant, CONTINUE



def train_sac2(config):
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    replay_buffer, replay_loader = get_buffer_loader()

    board = Board()
    board.reset()
    en_passant = None

    # 16x8x8 state
    white_state, black_state, next_state = board.get_state(), None, None
    white_action, black_action = None, None

    train_agent = SAC()
    fixed_agent = SAC()

    white_agent, black_agent = train_agent, fixed_agent

    current_team = WHITE        

    game_step = 0

    # ensures that the black state transition is not added after first move
    # and so that the final game runs to completion
    first_step = True

    while game_step < config.max_timesteps or not first_step:
        # white move
        if current_team == WHITE:
            next_state, next_white_action, en_passant, move_result = take_action(board, white_agent, current_team, en_passant)
        else:
            next_state, next_black_action, en_passant, move_result = take_action(board, white_agent, current_team, en_passant)
        
        game_step += 1

        # game continues
        if move_result == CONTINUE:
            #  if it was white's turn, then we should have a new state -> next_state transition for black
            # but only if it is not the first move in the game, since there is otherwise not a previous state in the transition
            if current_team == WHITE and not first_step:  
                replay_buffer.insert(black_state, next_state, black_action, 0, BLACK)
            #  if it was black's turn, then we should have a new state -> next_state transition for white
            elif current_team == BLACK:
                replay_buffer.insert(white_state, next_state, white_action, 0, WHITE)
        
        # if the game has ended
        # only loss/draw because it checks if the player has no moves left
        elif move_result in [LOSS, DRAW]:
            # draw (-0.25, -0.25), or white loss (-1, 1), or black loss (1, -1)
            # draw has slight penalty to discourage
            white_reward, black_reward = \
                (-0.25, -0.25) if move_result == DRAW else \
                (-1, 1) if current_team == WHITE else \
                (1, -1)
            
            replay_buffer.insert(white_state, next_state, white_action, white_reward, WHITE)
            replay_buffer.insert(black_state, next_state, black_action, black_reward, BLACK)
            
            # reset board and switch teams
            board.reset()
            first_step = True
            white_agent, black_agent = black_agent, white_agent
            current_team = WHITE

            continue

        if current_team == WHITE:
            white_state, white_action = next_state, next_white_action
        else:
            black_state, black_action = next_state, next_black_action
        
        current_team = other_team(current_team)
        first_step = False

        



def train_sac(config):
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    replay_buffer, replay_loader = get_buffer_loader()

    board = Board()
    board.reset()
    en_passant = None

    # 16x8x8 state
    white_board_state, black_board_state = board.get_state(), None
    white_action, black_action = None, None
    # map of piece -> list of moves
    move_map = get_moves(board, WHITE, en_passant)
    move_matrix = get_move_matrix(move_map)  # matrix representation

    train_agent = SAC()
    fixed_agent = SAC()

    white_agent, black_agent = train_agent, fixed_agent

    for step in range(config.max_timesteps):
        # sample white select and target action from agent as tuple
        white_action = white_agent.sample_action(white_board_state, move_matrix, WHITE)

        # unravel to coordinate representation
        white_select = np.unravel_index(white_action[0], (8, 8))
        white_target = np.unravel_index(white_action[1], (8, 8))

        # select and move piece
        selected_piece = board.coord_map[*white_select]
        en_passant = board.move_piece(selected_piece, Move(*white_target))

        # black board state and 
        next_black_board_state = board.get_state()
        move_map = get_moves(board, BLACK, en_passant)

        # white check mate or draw
        if not move_map:
            # white check mate
            if move_map is None:
                replay_buffer.insert(white_board_state, white_action, 1, next_black_board_state)
                replay_buffer.insert(black_board_state, black_action, -1, next_black_board_state)
            
            # draw, slight penalty to avoid it
            else:
                replay_buffer.insert(white_board_state, white_action, -0.25, next_black_board_state)
                replay_buffer.insert(black_board_state, black_action, -0.25, next_black_board_state)

            # reset board and switch teams
            board.reset()
            en_passant = None
            white_agent, black_agent = black_agent, white_agent
            black_action = None  # so that black tuple is not added in first game iteration

            # initial state and moves, and restard loop
            white_board_state = board.get_state()
            move_map = get_moves(board, WHITE, en_passant)
            continue

        # game continues
        elif black_action is not None:
            replay_buffer.insert(black_board_state, black_action, 0, next_black_board_state)
        
        # sample white select and target action from agent as tuple
        black_board_state = next_black_board_state
        move_matrix = get_move_matrix(move_map)
        black_action = black_agent.sample_action(black_board_state, move_matrix)

        # unravel to coordinate representation
        black_select = np.unravel_index(black_action[0], (8, 8))
        black_target = np.unravel_index(black_action[1], (8, 8))

        # select and move piece
        selected_piece = board.coord_map[*black_select]
        en_passant = board.move_piece(selected_piece, Move(*black_target))
        
        next_white_board_state = board.get_state()
        move_map = get_moves(board, WHITE, en_passant)

