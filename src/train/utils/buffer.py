import torch
from torch.utils.data import Dataset, DataLoader
from collections import deque


class ReplayBuffer(Dataset):
    def __init__(self, capacity = 10000):
        self.buffer = deque(maxlen=capacity)

    def insert(self, state, next_state, action, reward):
        self.buffer.append((state, action, reward, next_state))

    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, idx):
        return list(map(torch.tensor, self.buffer[idx]))


def get_buffer_loader(capacity = 10000, batch_size = 64, shuffle=True):
    replay_buffer = ReplayBuffer(capacity)
    dataloader = DataLoader(
        replay_buffer,
        batch_size=batch_size,
        shuffle=shuffle)
    
    return replay_buffer, dataloader