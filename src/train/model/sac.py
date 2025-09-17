from .common import Actor, DoubleCritic, StateEncoder
from game.constants import Team
import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam
from torch import nn
from typing import Tuple, List
from train.model.buffer import Batch
from train.utils import validate_tensors


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
            lr = 1e-4,
            tau = 0.005):
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr
        self.tau = tau

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = StateEncoder().to(device=self.device)
        self.actor = Actor().to(device=self.device)
        self.critic = DoubleCritic().to(device=self.device)
        self.target_critic = DoubleCritic().to(device=self.device)

        # trainable entropy magnitude parameter
        self.log_alpha = nn.Parameter(torch.tensor(0.0))  
        self.target_entropy = -act_dim

        self.log_alpha_opt = Adam([self.log_alpha], lr = lr)
        self.actor_opt = Adam(list(self.encoder.parameters()) + list(self.actor.parameters()), lr = lr)
        self.critic_opt = Adam(list(self.encoder.parameters()) + list(self.critic.parameters()), lr = lr)

        self.alpha_loss_func = AlphaLoss()
        self.actor_loss_func = ActorLoss()
        self.mse_loss_func = nn.MSELoss()
    

    def state_dict(self):
        return {
            "encoder_params" : self.encoder.state_dict(),
            "actor_params" : self.actor.state_dict(),
            "critic_params" : self.critic.state_dict()}
    
    def load_state_dict(self, state_dict : dict):
        self.encoder.load_state_dict(state_dict["encoder_params"])
        self.actor.load_state_dict(state_dict["actor_params"])
        self.critic.load_state_dict(state_dict["critic_params"])
    
    def train(self):
        self.encoder.train()
        self.actor.train()
        self.critic.train()
    
    def eval(self):
        self.encoder.eval()
        self.actor.eval()
        self.critic.eval()

    
    def sample_actions(self, board_states, move_matrices, teams, eval = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Sample one or more actions from agent.

        board_states : Bx16x8x8 tensor (B = batch size)
        move_matrices : Bx64x64 tensor, 
        - row is selection position
        - col is target position
        - 0 invalid, 1 valid, 2 valid and pawn promotion

        Returns: select, target, promote, logp
        """
        board_states, move_matrices, teams = validate_tensors([board_states, move_matrices, teams], [4, 3, 2])
        
        if eval:
            self.eval()
        else:
            self.train()

        embedding = self.encoder(board_states)
        return self.actor(embedding, teams, move_matrices)


    def actor_alpha_train_step(self, batch : Batch):

        self.log_alpha_opt.zero_grad()
        self.actor_opt.zero_grad()
        
        embeddings = self.encoder(batch.states)

        # sample actions from given states
        select, target, promote, logp = self.actor.forward(embeddings, batch.teams, batch.move_matrices)

        # train alpha parameter
        alpha = torch.exp(self.log_alpha)
        alpha_loss = self.alpha_loss_func(alpha, logp.detach(), self.target_entropy)
        alpha_loss.backward()

        # train actor (critic parameters are detached)
        q1, q2 = self.critic.forward(embeddings, batch.teams, select, target, promote)
        sampled_q = torch.minimum(q1, q2).detach()
        actor_loss = self.actor_loss_func(alpha, logp, sampled_q)
        actor_loss.backward()

        self.log_alpha_opt.step()
        self.actor_opt.step()

        log_info = {
            "alpha_loss": alpha_loss.detach(),
            "actor_loss": actor_loss.detach(),
            "alpha": alpha,
            "action_logp": logp.detach() 
        }

        return log_info
    
    def critic_train_step(self, batch : Batch, alpha : Tensor):

        self.critic_opt.zero_grad()

        # episode termination flag
        done = (batch.rewards != 0).int()
        
        embeddings = self.encoder(batch.states)
        next_embeddings = self.encoder(batch.next_states)

        # current q
        q1, q2 = self.critic(embeddings, batch.teams, batch.selections, batch.targets, batch.promote)

        next_select, next_target, next_promote, action_logp = self.actor.forward(next_embeddings, batch.next_move_matrices)

        with torch.no_grad():
            next_q1, next_q2 = self.target_critic.forward(next_embeddings, batch.teams, next_select, next_target, next_promote)
            next_q = torch.minimum(next_q1, next_q2) - alpha * action_logp.reshape(-1, 1)

            # target q, filtered by whether the next state is terminating
            target_q = batch.rewards + self.gamma * (1 - done) * next_q.reshape(-1, 1)

        critic_loss1 = self.mse_loss_func(q1, target_q)
        critic_loss2 = self.mse_loss_func(q2, target_q)
        critic_loss = (critic_loss1 + critic_loss2).mean()

        critic_loss.backward()

        self.critic_opt.step()

        # update target critic parameters
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        log_info = {
            "critic_loss": critic_loss.detach(),
            "q": q1
        }

        return log_info

        
    def train_step(self, batch : Batch):
        self.train()

        self.optimizer.zero_grad()

        actor_log_info = self.actor_alpha_train_step(batch)
        critic_log_info = self.critic_train_step(batch, actor_log_info["alpha"])

        self.optimizer.step()

        log_info = { **actor_log_info, **critic_log_info }

        return log_info