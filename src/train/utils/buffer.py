import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from collections import deque
from game.constants import Team
from collections import namedtuple
import numpy as np


batch_attributes = [
    "states", 
    "next_states", 
    "selections", 
    "targets", 
    "move_matrices",
    "rewards",
    "teams"]


Batch = namedtuple("Batch", batch_attributes)


class ReplayBuffer(Dataset):
    def __init__(self, capacity = 10000, batch_size = 128):
        self.buffer = deque(maxlen=capacity)
        self.device = ["cuda" if torch.cuda.is_available() else "cpu"]
        self.batch_size = batch_size

    def insert(self, state, next_state, select, target, move_matrix, reward, team):
        if not isinstance(state, Tensor):
            state = torch.tensor(state)
        if not isinstance(next_state, Tensor):
            next_state = torch.tensor(next_state)
        if not isinstance(action, Tensor):
            action = torch.tensor(action)
        if not isinstance(move_matrix, Tensor):
            move_matrix = torch.tensor(move_matrix)
        if not isinstance(reward, Tensor):
            reward = torch.tensor(reward)
        if not isinstance(team, Tensor):
            team = torch.tensor(team.value if isinstance(team, Team) else team)

        self.buffer.append((state, next_state, select, target, move_matrix, reward, team))

    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, idx):
        return self.buffer[idx]
    
    def sample_batch(self) -> Batch:
        idxs = np.random.choice(np.arange(len(self.buffer)), size = self.batch_size, replace = False)
        # list of tuples (state, next_state, select, ...)
        samples = [self.buffer[idx] for idx in idxs]
        # unzip and convert to tensors (states, next_states, selections, ...)
        tensors = map(torch.stack, zip(*samples))
        # upload to device
        tensors = list(map(lambda tensor: tensor.to(device=self.device), tensors))
        
        return Batch(*tensors)