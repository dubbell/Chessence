import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from torch.distributions import Categorical

from collections import namedtuple
from train.utils import tensor_check, to_ndim


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
    """Takes board state and outputs 512 dim embedding, to be concatenated with team."""

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            SimpleConv(16, 32, False),  # outputs 7x7
            SimpleConv(32, 32, False), # outputs 6x6 
            SimpleConv(32, 32, False), # outputs 5x5 
            SimpleConv(32, 64, False), # outputs 4x4 
            SimpleConv(64, 64, False), # outputs 3x3
            SimpleConv(64, 128, False), # outputs 2x2
            nn.Conv2d(128, 512, kernel_size=2, stride=1), # outputs 1x1
            nn.Flatten())
    
    def forward(self, board_states : Tensor) -> Tensor:
        board_states = to_ndim(tensor_check(board_states), 4)
        assert board_states.shape[1:] == (16, 8, 8)

        return self.conv(board_states)


class Critic(nn.Module):
    """Action-value function, takes board embedding (512) and produces estimated value."""
    
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
        
    
    def forward(self, embeddings : Tensor, select : Tensor, target : Tensor, promote : Tensor):
        embeddings, select, target, promote = map(tensor_check, [embeddings, select, target, promote])

        assert embeddings.shape[1:] == (512,), embeddings.shape
        assert select.dim() == 1, select.shape
        assert target.dim() == 1, target.shape
        assert promote.dim() == 1, promote.shape

        one_hot_select = F.one_hot(select, 64)
        one_hot_target = F.one_hot(target, 64)
        one_hot_promote = torch.zeros((promote.shape[0], 4)).long()
        one_hot_promote[promote != -1] = F.one_hot(promote[promote != -1], 4)
        stacked = torch.hstack((embeddings, one_hot_select, one_hot_target, one_hot_promote))
        return self.linear(stacked).reshape(-1)


class DoubleCritic(nn.Module):
    """Double critic implementation to account for overestimation bias in Q-values."""

    def __init__(self):
        super().__init__()

        self.critic1 = Critic()
        self.critic2 = Critic()

    def forward(self, board_embs : Tensor, select : Tensor, target : Tensor, promote : Tensor):
        return self.critic1(board_embs, select, target, promote), self.critic2(board_embs, select, target, promote)
        


Action = namedtuple("Action", ["select", "target", "promote", "logp"])


class Actor(nn.Module):

    def __init__(self):
        super().__init__()

        self.exploring = False

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
            nn.Linear(512, 64))
        
        self.targeter = nn.Sequential(
            nn.Linear(576, 512),
            nn.ReLU(),
            nn.Linear(512, 64))
        
        self.promoter = nn.Sequential(
            nn.Linear(640, 512),
            nn.ReLU(),
            nn.Linear(512, 4))
        
    def explore(self):
        self.exploring = True

    def exploit(self):
        self.exploring = False
    
    def get_select(self, projected, move_matrices):
        assert projected.shape[1:] == (512,), projected.shape
        assert move_matrices.shape[1:] == (64, 64), move_matrices.shape

        select_filter = move_matrices.any(axis=2)
        if (select_filter == 0).all():
            select_logits = torch.zeros_like(select_filter).float()
        else:
            select_logits = self.selector(projected)
            select_logits.masked_fill_(~select_filter, float('-inf'))

        if self.exploring:
            distr = Categorical(logits=select_logits)
            select = distr.sample()
            logp = distr.log_prob(select)
        else:
            select = select_logits.argmax(axis=1).long()
            logp = F.log_softmax(select_logits, dim=1)[torch.arange(select.shape[0]), select]

        return select, logp
    
    def get_target(self, projected, move_matrices, select):
        assert projected.shape[1:] == (512,), projected.shape
        assert move_matrices.shape[1:] == (64, 64), move_matrices.shape
        assert select.dim() == 1, select.shape

        target_filter = move_matrices[torch.arange(select.shape[0]), select] > 0
        if (target_filter == 0).all():
            target_logits = torch.zeros_like(target_filter).float()
        else:
            target_logits = self.targeter(torch.hstack((projected, F.one_hot(select, 64))))
            target_logits.masked_fill_(~target_filter, float('-inf'))

        if self.exploring:
            distr = Categorical(logits=target_logits)
            target = distr.sample()
            logp = distr.log_prob(target)
        else:
            target = target_logits.argmax(axis=1).long()
            logp = F.log_softmax(target_logits, dim=1)[torch.arange(target.shape[0]), target]

        return target, logp

    def get_promote(self, projected, move_matrices, select, target):
        assert projected.shape[1:] == (512,), projected.shape
        assert move_matrices.shape[1:] == (64, 64), move_matrices.shape
        assert select.dim() == 1, select.shape
        assert target.dim() == 1, target.shape
               
        promote_filter = move_matrices[torch.arange(select.shape[0]), select, target] == 2
        promote_logits = self.promoter(torch.hstack((projected, F.one_hot(select, 64), F.one_hot(target, 64))))
        if self.exploring:
            distr = Categorical(logits=promote_logits)
            promote = distr.sample()
            promote_logp = distr.log_prob(promote)
        else:
            promote = promote_logits.argmax(axis=1).long()
            promote_logp = torch.log_softmax(promote_logits, dim=1)[torch.arange(promote.shape[0]), promote]

        promote.masked_fill_(~promote_filter, -1)
        promote_logp.masked_fill_(~promote_filter, 0)

        return promote, promote_logp

    
    def forward(self, embeddings : Tensor, move_matrices : Tensor):
        embeddings, move_matrices = map(tensor_check, [embeddings, move_matrices])

        assert embeddings.shape[1:] == (512,), embeddings.shape
        assert move_matrices.shape[1:] == (64, 64), move_matrices.shape

        projected = self.body(embeddings)

        select, select_logp = self.get_select(projected, move_matrices)
        target, target_logp = self.get_target(projected, move_matrices, select)
        promote, promote_logp = self.get_promote(projected, move_matrices, select, target)

        return select, target, promote, select_logp + target_logp + promote_logp

        