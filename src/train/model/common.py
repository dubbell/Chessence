import torch.nn as nn
import torch
from torch import Tensor
from game.constants import Team, BLACK


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
    """Value function, takes team and board embedding (1 and 511) and produces estimated value."""
    
    def __init__(self):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(512, 512), 
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512), 
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1))
        
    
    def forward(self, board_emb : Tensor, team : Team):
        team_num = (-1 if team == BLACK else 1)
        torch.concat((board_emb, team_num))
        return self.linear(board_emb).squeeze()
        

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
            nn.Softmax(64))
        
        self.targeter = nn.Sequential(
            nn.Linear(512, 64),
            nn.Softmax(64))
    
    def forward(self, x : Tensor):
        x = self.body(x)
        select_distr = self.selector(x)
        target_distr = self.targeter(x)

        return select_distr, target_distr