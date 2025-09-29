import torch
from torch import Tensor
from torch.utils.data import Dataset
from game.constants import Team
from collections import namedtuple
import numpy as np
from train.utils import tensor_check


batch_attributes = [
    "states", 
    "next_states", 
    "move_matrices",
    "next_move_matrices",
    "select", 
    "target", 
    "promote",
    "rewards",
    "done"]


Batch = namedtuple("Batch", batch_attributes)


class ReplayBuffer(Dataset):
    def __init__(self, capacity = 10000, batch_size = 128):
        self.capacity = capacity
        self.buffer = [None for _ in range(capacity)]
        self.current_idx = 0
        self.length = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size


    def insert(self, state, next_state, move_matrix, next_move_matrix, select, target, promote, reward, done):
        self.buffer[self.current_idx] = list(map(tensor_check, [state, next_state, move_matrix, next_move_matrix, select, target, promote, reward, done]))
        self.current_idx = (self.current_idx + 1) % self.capacity
        self.length = min(self.capacity, self.length + 1)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """Sampled tuple."""
        idx = idx % self.length
        return (*self.buffer[idx],)

    def stack_tensor(self, values):
        return torch.stack(values).squeeze().to(device=self.device)

    def sample_batch(self) -> Batch:
        idxs = np.random.choice(np.arange(self.length), size = self.batch_size, replace = False)
        # list of tuples (state, next_state, move_matrix, ...)
        samples = [self.__getitem__(idx) for idx in idxs]
        # unzip and convert to tensors (states, next_states, move_matrices, ...)
        tensors = map(self.stack_tensor, zip(*samples))
        
        return Batch(*tensors)