from .common import Actor, DoubleCritic, StateEncoder
from game.constants import Team
import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam
from torch import nn
from typing import Tuple
from train.utils.buffer import Batch


class AlphaLoss(nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, alpha, logp, target_entropy):
        alpha_loss = -alpha * (logp + target_entropy)
        return alpha_loss.mean()


class ActorLoss(nn.Module):
    def __init__(self):
        super(ActorLoss, self).__init__()

    def forward(self, alpha, logp, sampled_q):
        actor_loss = alpha * logp - sampled_q
        return actor_loss.mean()


class SAC:
    """Soft Actor-Critic implementation."""

    def __init__(
            self,
            obs_dim = 512,
            act_dim = 128,
            gamma = 0.99,
            lr = 1e-4):
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = StateEncoder().to(device=self.device)
        self.actor = Actor().to(device=self.device)
        self.critic = DoubleCritic().to(device=self.device)

        self.log_alpha = torch.tensor(0.0, requires_grad = True)  # trainable entropy magnitude parameter
        self.target_entropy = -act_dim

        self.optimizer = Adam(lr = lr)
        self.alpha_loss_func = AlphaLoss()
        self.actor_loss_func = AlphaLoss()
        self.mse_loss_func = nn.MSELoss()
        
    
    def train(self):
        self.encoder.train()
        self.actor.train()
        self.critic.train()
    
    def eval(self):
        self.encoder.eval()
        self.actor.eval()
        self.critic.eval()
    
    
    def get_action_samples(self, select_distrs : Tensor, target_distrs : Tensor, move_matrices : Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Get sampled action given select and target distributions, along with the available moves in a 64x64 move matrix.
           Returns selected square, target square, and log likelihood of the total action."""
        if not isinstance(select_distrs, Tensor):
            select_distrs = torch.tensor(select_distrs)
        if not isinstance(target_distrs, Tensor):
            target_distrs = torch.tensor(target_distrs)
        if not isinstance(move_matrices, Tensor):
            move_matrices = torch.tensor(move_matrices)
        
        select_filter : Tensor = move_matrices.any(axis=2).to(torch.int16)
        filtered_select_distrs = select_filter * select_distrs
        select = filtered_select_distrs.argmax(axis=1)

        target_filter : Tensor = move_matrices[np.arange(len(move_matrices)), select]
        filtered_target_distrs = target_filter * target_distrs
        target = filtered_target_distrs.argmax(axis=1)

        # log probability after normalizing probabilities to account for filtering
        select_logp = torch.log(filtered_select_distrs[np.arange(len(select_distrs)), select] / filtered_select_distrs.sum())
        target_logp = torch.log(filtered_target_distrs[np.arange(len(target_distrs)), target] / filtered_target_distrs.sum())

        return select, target, select_logp + target_logp

    
    def sample_action(self, board_state, move_matrix, team : Team, eval = False):
        """
        Sample single action.

        board_state : Bx16x8x8 matrix (B = batch size)
        move_matrix : 64x64 matrix, 
        - row is selection position
        - col is target position
        - [row, col] determines if generated move is valid
        """
        if eval:
            self.eval()
        else:
            self.train()

        embedding = self.encoder(board_state)
        select_distr, target_distr = self.actor(embedding, [team])

        select, target, _ = self.get_action_samples(select_distr, target_distr, move_matrix)

        return select, target
    

    def actor_alpha_train_step(self, batch : Batch):
        
        self.train()

        self.optimizer.zero_grad()

        embeddings = self.encoder(batch.states)

        # sample actions from given states
        select_distrs, target_distrs = self.actor(embeddings, batch.teams)
        selections, targets, logp = self.get_action_samples(select_distrs, target_distrs, batch.move_matrices)

        # train alpha parameter
        alpha = torch.exp(self.log_alpha)
        alpha_loss = self.alpha_loss_func(alpha, logp.detach(), self.target_entropy)
        alpha_loss.backward()
        alpha = alpha.detach()

        # train actor (critic parameters are detached)
        q1, q2 = self.critic(embeddings, batch.teams, selections, targets)
        sampled_q = torch.minimum(q1, q2).detach()
        actor_loss = self.actor_loss_func(alpha, logp, sampled_q)
        actor_loss.backward()

        self.optimizer.step()

        log_info = {
            "alpha_loss": alpha_loss.detach(),
            "actor_loss": actor_loss.detach(),
            "alpha": alpha,
            "action_logp": logp.detach() 
        }

        return log_info
    
    def critic_train_step(self, batch : Batch, alpha : Tensor):
        
        self.train()

        self.optimizer.zero_grad()

        embeddings = self.encoder(batch.states)
        next_embeddings = self.encoder(batch.next_states)

        # current q
        q1, q2 = self.critic(embeddings, batch.teams, batch.selections, batch.targets)

        next_select_distrs, next_target_distrs = self.actor(next_embeddings, batch.teams)
        next_select, next_target, action_logp = self.get_action_samples(next_select_distrs, next_target_distrs, batch.next_move_matrices)

        next_q1, next_q2 = self.critic(next_embeddings, batch.teams, next_select.detach(), next_target.detach())
        next_q = torch.minimum(next_q1, next_q2) - alpha * action_logp.detach()

        target_q = batch.rewards + self.gamma * next_q

        critic_loss1 = self.mse_loss_func(q1, target_q)
        critic_loss2 = self.mse_loss_func(q2, target_q)
        critic_loss = (critic_loss1 + critic_loss2).mean()

        critic_loss.backward()

        self.optimizer.step()

        log_info = {
            "critic_loss": critic_loss.detach(),
            "q": q1
        }

        return log_info

        
    def train_step(self, batch : Batch):
        actor_log_info = self.actor_alpha_train_step(batch)
        critic_log_info = self.critic_train_step(batch, actor_log_info["alpha"])

        log_info = { **actor_log_info, **critic_log_info }

        return log_info