from ..utils.buffer import ReplayBuffer
from game.move_calc import get_moves
from game.model import Board, Move
from game.constants import *
from game.utils import other_team
from ..model.sac import SAC

import git
import os
import numpy as np
import mlflow
from datetime import datetime
from tqdm import tqdm

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
    Output: next state, selection, target, move matrix, potential en passant location
    """

    current_state = board.get_state()
    move_map = get_moves(board, team, en_passant)
    # no moves available because checkmate
    if move_map is None:
        return current_state, None, None, None, LOSS
    # no moves available because draw
    elif not move_map:
        return current_state, None, None, None, DRAW

    move_matrix = get_move_matrix(move_map)
    action = agent.sample_action(current_state, move_matrix, team.value, eval=True)

    select = np.concatenate(np.unravel_index(action[0], (8, 8)))
    target = np.concatenate(np.unravel_index(action[1], (8, 8)))

    selected_piece = board.coord_map[*select]
    en_passant = board.move_piece(selected_piece, Move(target))
    
    next_state = board.get_state()

    return next_state, action, en_passant, move_matrix, CONTINUE


def get_run_name():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"SAC_{timestamp}"


def get_latest_commit_hash():
    repo = git.Repo(".", search_parent_directories=True)
    return repo.head.commit.hexsha

    
def start_training(config):
    replay_buffer = ReplayBuffer()

    board = Board()
    board.reset()
    en_passant = None

    # 16x8x8 state
    white_state, black_state, next_state = board.get_state(), None, None
    white_action, black_action = None, None
    white_move_matrix, black_move_matrix = None, None

    train_agent, fixed_agent = SAC(), SAC()
    white_agent, black_agent = train_agent, fixed_agent

    current_team = WHITE

    step = 0
    game_count = 0

    # ensures that the black state transition is not added after first move
    # and so that the final game runs to completion
    first_step = True

    if config["enable_logging"]:
        mlflow.log_params({
            "learning_rate": train_agent.lr,
            "gamma": train_agent.lr,
            "commit": get_latest_commit_hash(),
            **config})
        
    step_size = 0.0001
    next_pb_step = step_size
    pb = tqdm(total = 1 / step_size)
    
    while step < config["max_timesteps"] or not first_step:
        if step / config["max_timesteps"] >= next_pb_step:
            next_pb_step += step_size
            pb.update()

        # train non-fixed agent
        if step >= config["train_start"] and step % config["train_interval"] == 0:
            train_agent.train_step(replay_buffer.sample_batch())

        # update fixed agent parameters to trained agent parameters
        if step % config["update_interval"]:
            fixed_agent.load_state_dict(train_agent.state_dict())

        # white move
        if current_team == WHITE:
            next_state, next_white_action, en_passant, next_white_move_matrix, move_result = take_action(board, white_agent, current_team, en_passant)
        else:
            next_state, next_black_action, en_passant, next_black_move_matrix, move_result = take_action(board, white_agent, current_team, en_passant)
        
        step += 1

        # game continues
        if move_result == CONTINUE:
            #  if it was white's turn, then we should have a new state -> next_state transition for black
            # but only if it is not the first move in the game, since there is otherwise not a previous state in the transition
            if current_team == WHITE and not first_step:  
                replay_buffer.insert(black_state, next_state, *black_action, black_move_matrix, 0, BLACK)
            #  if it was black's turn, then we should have a new state -> next_state transition for white
            elif current_team == BLACK:
                replay_buffer.insert(white_state, next_state, *white_action, white_move_matrix, 0, WHITE)
        
        # if the game has ended, by checking if the current player has no moves left
        elif move_result in [LOSS, DRAW]:
            # draw (-0.1, -0.1), or white loss (-1, 1), or black loss (1, -1)
            # draw has slight penalty to discourage
            white_reward, black_reward = \
                (-0.1, -0.1) if move_result == DRAW else \
                (-1, 1) if current_team == WHITE else \
                (1, -1)
            
            replay_buffer.insert(white_state, next_state, *white_action, white_move_matrix, white_reward, WHITE)
            replay_buffer.insert(black_state, next_state, *black_action, black_move_matrix, black_reward, BLACK)
            
            # reset board and switch teams
            board.reset()
            first_step = True
            white_state = board.get_state()
            white_agent, black_agent = black_agent, white_agent
            current_team = WHITE

            if config["enable_logging"]:
                is_white = game_count % 2 == 0  # if train_agent is white
                mlflow.log_metric(
                    "white_win" if is_white else "black_win",
                    int(is_white != (current_team == WHITE)))

            game_count += 1

            continue

        if current_team == WHITE:
            white_state, white_action, white_move_matrix = next_state, next_white_action, next_white_move_matrix
        else:
            black_state, black_action, black_move_matrix = next_state, next_black_action, next_black_move_matrix
        
        current_team = other_team(current_team)
        first_step = False
    
    pb.close()


def train_sac(config):
    if config["enable_logging"]:
        with mlflow.start_run(run_name=get_run_name(), log_system_metrics=True):
            start_training(config)
    else:
        start_training(config)