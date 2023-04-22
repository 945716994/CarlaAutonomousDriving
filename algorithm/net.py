import torch
import torch.nn as nn
import torch.nn.functional as F
from util import init_layers


class PolicyNetContinuous(torch.nn.Module):
    """用于连续动作空间的策略网络(PPO)"""

    def __init__(self, state_dim, hidden_dim, output_dim, hidden_layers):
        super(PolicyNetContinuous, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('input_layer', nn.Linear(state_dim, hidden_dim))
        self.net.add_module('activation_layer', nn.ReLU())
        for i in range(hidden_layers):
            self.net.add_module('hidden_layer', nn.Linear(hidden_dim, hidden_dim))
            self.net.add_module('activation_layer', nn.ReLU())
        self.fc_mu = torch.nn.Linear(hidden_dim, output_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.net(x)
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        # softplus : log(1+e^x) 是平滑版的RELU
        std = F.softplus(self.fc_std(x))
        return mu, std


class PolicyNet(torch.nn.Module):
    """PPO可用的策略网络"""

    def __init__(self, state_dim, hidden_dim, output_dim, hidden_layers):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('input_layer', nn.Linear(state_dim, hidden_dim))
        self.net.add_module('activation_layer', nn.ReLU())
        for i in range(hidden_layers):
            self.net.add_module('hidden_layer', nn.Linear(hidden_dim, hidden_dim))
            self.net.add_module('activation_layer', nn.ReLU())
        self.net.add_module(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        x = self.net(x)
        # 用softmax得到动作分布 P(a|s) --(每个动作被选择的相应概率)
        return F.softmax(x, dim=1)


class ValueNet(torch.nn.Module):
    """Actor-critic中的critic Net"""

    def __init__(self, state_dim, hidden_dim, output_dim, hidden_layers):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('input_layer', nn.Linear(state_dim, hidden_dim))
        self.net.add_module('activation_layer', nn.ReLU())
        for i in range(hidden_layers):
            self.net.add_module('hidden_layer', nn.Linear(hidden_dim, hidden_dim))
            self.net.add_module('activation_layer', nn.ReLU())
        self.net.add_module(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    policy = PolicyNetContinuous(30, 32, 5, 3)
    init_layers(policy.net)
