import torch.nn as nn
import torch
from torch import Tensor
from game.constants import Team, BLACK
from typing import List


class SimpleConv(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, keep_size : bool):
        super().__init__()

        self.keep_size = keep_size

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=1)
        self.dropout = nn.Dropout(p=0.2)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x : Tensor):
        if self.keep_size:
            # add padding to keep input and output same size
            x = self.conv(nn.functional.pad(x, [0, 1, 0, 1]))
        else:
            x = self.conv(x)
        
        return self.relu(self.batch_norm(self.dropout(x)))
        

class StateEncoder(nn.Module):
    """Takes board state and outputs 511 dim embedding, to be concatenated with team."""

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            SimpleConv(16, 32, False),  # outputs 7x7
            SimpleConv(32, 32, False), # outputs 6x6 
            SimpleConv(32, 32, False), # outputs 5x5 
            SimpleConv(32, 64, False), # outputs 4x4 
            SimpleConv(64, 64, False), # outputs 3x3
            SimpleConv(64, 128, False), # outputs 2x2
            nn.Conv2d(128, 511, kernel_size=2, stride=1), # outputs 1x1
            nn.Flatten())
    
    def forward(self, board_state : Tensor) -> Tensor:
        if not isinstance(board_state, torch.Tensor):
            board_state = torch.tensor(board_state)
        return self.conv(board_state)


class Critic(nn.Module):
    """Action-value function, takes team and board embedding (1 and 511) and produces estimated value."""
    
    def __init__(self):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(640, 512), 
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512), 
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1))
        
    
    def forward(self, board_embs : Tensor, teams : Tensor, selections : Tensor, targets : Tensor):
        one_hot_select = nn.functional.one_hot(selections, 64)
        one_hot_target = nn.functional.one_hot(targets, 64)
        stacked = torch.hstack((board_embs, teams, one_hot_select, one_hot_target))
        return self.linear(stacked).squeeze()


class DoubleCritic(nn.Module):
    """Double critic implementation to account for overestimation bias in Q-values."""

    def __init__(self):
        super().__init__()

        self.critic1 = Critic()
        self.critic2 = Critic()

    def forward(self, board_embs : Tensor, teams : Tensor, selections : Tensor, targets : Tensor):
        return self.critic1(board_embs, teams, selections, targets), self.critic2(board_embs, teams, selections, targets)
        
        

class Actor(nn.Module):

    def __init__(self):
        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(512, 512), 
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512), 
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512))

        self.selector = nn.Sequential(
            nn.Linear(512, 64),
            nn.Softmax(1))
        
        self.targeter = nn.Sequential(
            nn.Linear(512, 64),
            nn.Softmax(1))
    
    def forward(self, board_embs : Tensor, teams : Tensor):
        stacked = torch.hstack((board_embs, teams))
        projected = self.body(stacked)

        return self.selector(projected), self.targeter(projected)