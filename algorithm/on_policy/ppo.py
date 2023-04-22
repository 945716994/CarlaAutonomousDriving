import random
import gymnasium as gym
import torch
from torch.distributions.normal import Normal
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..util import init_layers
from ..net import PolicyNet, ValueNet, PolicyNetContinuous


class PPO(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device,
                 action_type="continuous", hidden_layers=3):
        super(PPO, self).__init__()
        self.critic = ValueNet(state_dim, hidden_dim, action_dim, hidden_layers).to(device)
        if action_type == "continuous":
            self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, hidden_layers).to(device)
        else:
            self.actor = PolicyNet(state_dim, hidden_dim, action_dim, hidden_layers).to(device)

        # 初始化网络参数
        init_layers(self.actor)
        init_layers(self.critic)
        # 设置网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        # 一条序列的数据用来训练轮数
        self.epochs = epochs
        # PPO-clip中截断范围的参数
        self.eps = eps
        self.device = device
        self.action_type = action_type

    def take_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        elif isinstance(state, torch.Tensor):
            state = state.to(self.device)
        else:
            raise TypeError("state must be either a numpy array or a PyTorch tensor")

        if self.action_type == "continuous":
            action_mean, action_std = self.actor(state)
            probs = Normal(action_mean, action_std)
            action = probs.sample()
        else:
            probs = self.actor(state)
            action_list = torch.distributions.Categorical(probs)
            action = action_list.sample()
        return action.cpu().detach().numpy()

    def update(self, transition):

        pass

    # Define the PPO save function
    def save(self, path):
        # TODO: Implement PPO save algorithm
        pass

    # Define the PPO load function
    def load(self, path):
        # TODO: Implement PPO load algorithm
        pass
