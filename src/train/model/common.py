import torch.nn as nn
from torch.nn.functional import one_hot
import torch
from torch import Tensor
from collections import namedtuple
from train.utils import tensor_check


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
        board_state = tensor_check(board_state)
        if board_state.dim() == 3:
            board_state = board_state.unsqueeze(0)
        return self.conv(board_state)


class Critic(nn.Module):
    """Action-value function, takes team and board embedding (1 and 511) and produces estimated value."""
    
    def __init__(self):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(644, 512), 
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512), 
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1))
        
    
    def forward(self, embedding : Tensor, team : Tensor, select : Tensor, target : Tensor, promote : Tensor):
        embedding, team, select, target, promote = map(tensor_check, [embedding, team, select, target, promote])

        one_hot_select = one_hot(select.reshape(-1).long(), 64)
        one_hot_target = one_hot(target.reshape(-1).long(), 64)
        one_hot_promote = torch.zeros((len(promote), 4)).long()
        one_hot_promote[promote.squeeze() != -1] = one_hot(promote[promote != -1], 4)
        stacked = torch.hstack((embedding, team, one_hot_select, one_hot_target, one_hot_promote))
        return self.linear(stacked)


class DoubleCritic(nn.Module):
    """Double critic implementation to account for overestimation bias in Q-values."""

    def __init__(self):
        super().__init__()

        self.critic1 = Critic()
        self.critic2 = Critic()

    def forward(self, board_embs : Tensor, teams : Tensor, select : Tensor, target : Tensor, promote : Tensor):
        return self.critic1(board_embs, teams, select, target, promote), self.critic2(board_embs, teams, select, target, promote)
        


Action = namedtuple("Action", ["select", "target", "promote", "logp"])


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
            nn.ReLU())

        self.selector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 64), 
            nn.Softmax(1))
        
        self.targeter = nn.Sequential(
            nn.Linear(576, 512),
            nn.ReLU(),
            nn.Linear(512, 64), 
            nn.Softmax(1))
        
        self.promoter = nn.Sequential(
            nn.Linear(640, 512),
            nn.ReLU(),
            nn.Linear(512, 4), 
            nn.Softmax(1))
    
    def get_select(self, projected, move_matrices):
        select_filter = move_matrices.any(axis=2)
        select_distrs = select_filter * self.selector(projected)
        select_distrs /= torch.sum(select_distrs, dim=1, keepdim=True)
        select = select_distrs.argmax(axis=1, keepdim=True)
        logp = torch.log(select_distrs[torch.arange(len(select_distrs)), select])

        return select, logp
    
    def get_target(self, projected, move_matrices, select):
        target_filter = move_matrices[torch.arange(len(select)), select.squeeze()]
        target_distrs = target_filter * self.targeter(torch.hstack((projected, one_hot(select.reshape(-1), 64))))
        target_distrs /= torch.sum(target_distrs, axis=1, keepdim=True)
        target = target_distrs.argmax(axis=1, keepdim=True)
        logp = torch.log(target_distrs[torch.arange(len(select)), target.squeeze()])

        return target, logp

    def get_promote(self, projected, move_matrices, select, target):
        # initialize to -1, representing no promotion, with 0 logp
        promote = -torch.ones(len(select)).reshape(-1, 1).long()
        promote_logp = torch.zeros(len(select)).reshape(-1, 1).float()
        promote_filter = move_matrices[torch.arange(len(select)), select.squeeze(), target.squeeze()] == 2

        promote_distrs = self.promoter(
            torch.hstack((projected[promote_filter], one_hot(select[promote_filter].reshape(-1), 64), one_hot(target[promote_filter].reshape(-1), 64))))
        promote[promote_filter] = promote_distrs.argmax(axis=1, keepdim=True).long()
        promote_logp[promote_filter] = torch.log(torch.max(promote_distrs, axis=1, keepdim=True).values)

        return promote, promote_logp

    
    def forward(self, embeddings : Tensor, teams : Tensor, move_matrices : Tensor):
        embeddings, teams, move_matrices = map(tensor_check, [embeddings, teams, move_matrices])

        stacked = torch.hstack((embeddings, teams))
        projected = self.body(stacked)

        select, select_logp = self.get_select(projected, move_matrices)
        target, target_logp = self.get_target(projected, move_matrices, select)
        promote, promote_logp = self.get_promote(projected, move_matrices, select, target)

        return select, target, promote, select_logp + target_logp + promote_logp

        