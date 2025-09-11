from .common import Actor, Critic, StateEncoder
import numpy as np


class SAC:
    """Soft Actor-Critic implementation."""

    def __init__(
            self,
            obs_dim = 512,
            act_dim = 2,
            gamma = 0.99,
            lr = 1e-4):
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr

        self.encoder = StateEncoder()
        self.actor = Actor()
        self.critic = Critic()

    
    def sample_action(self, board_state, move_matrix):
        """
        Samples action. Returns None if no action is possible (shouldn't happen).

        board_state : 16x8x8 matrix
        move_matrix : 64x64 matrix, 
        - row is selection position
        - col is target position
        - [row, col] determines if generated move is valid
        """
        embedding = self.encoder(board_state)
        select_distr, target_distr = self.actor(embedding)

        # loop through selection coords in reverse order, i.e. highest selection value first
        for select_idx in select_distr.argsort()[::-1]:
            if move_matrix[select_idx].sum() == 0:
                continue

            for target_idx in target_distr.argsort()[::-1]:
                if not move_matrix[select_idx, target_idx]:
                    continue
                else:
                    return select_idx, target_idx
        
        return None