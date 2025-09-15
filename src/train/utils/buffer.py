import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from collections import deque
from game.constants import Team
from collections import namedtuple
import numpy as np
from train.experiments.train_sac import DRAW, LOSS


batch_attributes = [
    "states", 
    "next_states", 
    "move_matrices",
    "next_move_matrices",
    "selections", 
    "targets", 
    "rewards",
    "teams"]


Batch = namedtuple("Batch", batch_attributes)

def tensor_check(data):
    if not isinstance(data, Tensor):
        return torch.tensor(data)
    return data


class ReplayBuffer(Dataset):
    def __init__(self, capacity = 10000, batch_size = 128, draw_penalty = -0.1):
        self.capacity = capacity
        self.buffer = [None for _ in range(capacity)]
        self.current_idx = 0
        self.length = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.draw_penalty = draw_penalty


    def insert(self, state, move_matrix, select, target, reward, team):
        state, move_matrix, select, target, reward, team = \
            map(tensor_check, [state, move_matrix, select, target, reward, team.value if isinstance(team, Team) else team])

        self.buffer[self.current_idx] = (state, move_matrix, select, target, reward, team)
        self.current_idx = (self.current_idx + 1) % self.capacity
        self.length = min(self.capacity, self.length + 1)
    
    def set_win_rewards(self, move_result):
        """If game is terminated in current state, then set reward for previous moves to appropriate rewards."""
        first = (self.current_idx - 2) % self.length
        second = (self.current_idx - 1) % self.length
        # draw
        if move_result == DRAW:
            self.buffer[first][4] = self.draw_penalty
            self.buffer[second][4] = self.draw_penalty
        # loss
        else:
            self.buffer[first][4] = -1  # current player made the move two moves ago, so they are penalized
            self.buffer[second][4] = 1  # other player therefore won, and is rewarded
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """Sampled tuple."""
        idx = idx % self.length

        state, move_matrix, select, target, reward, team = self.buffer[idx]
        next_state, next_move_matrix = self.buffer[(idx+1) % self.length][:2]
            
        return state, next_state, move_matrix, next_move_matrix, select, target, reward, team


    def to_tensor(self, tuple):
        tensor = torch.stack(tuple).to(device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.reshape(-1, 1)
        return tensor
    
    def sample_batch(self) -> Batch:
        idxs = np.random.choice(np.arange(self.length - 1), size = self.batch_size, replace = False)
        # list of tuples (state, next_state, move_matrix, ...)
        samples = [self.__getitem__(idx) for idx in idxs]
        # unzip and convert to tensors (states, next_states, move_matrices, ...)
        tensors = map(self.to_tensor, zip(*samples))
        
        return Batch(*tensors)