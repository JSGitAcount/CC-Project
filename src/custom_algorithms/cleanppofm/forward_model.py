import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from torch.distributions import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProbabilisticSimpleForwardNet(nn.Module):
    def __init__(self, env, cfg):
        super().__init__()
        # FIXME:
        # hidden_size = cfg["hidden_size"]
        hidden_size = 256
        if isinstance(env.observation_space, spaces.dict.Dict):
            self.obs_shape = np.sum(
                [obs_space.shape for obs_space in env.observation_space.spaces.values()]
            )
        else:
            self.obs_shape = np.prod(env.observation_space.shape)
        self.action_shape = int(np.prod(env.action_space.shape))

        self.state_action_encoder = nn.Sequential(
            nn.Linear(self.obs_shape + self.action_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.fw_mu = nn.Linear(hidden_size, self.obs_shape)
        self.fw_std = nn.Linear(hidden_size, self.obs_shape)

    def forward(self, obs, action):
        # forward model: p(w' | w, a)
        hx = torch.cat([obs, action], dim=-1)
        hx = self.state_action_encoder(hx)
        hx = F.relu(hx)
        fw_mu, fw_log_std = self.fw_mu(hx), self.fw_std(hx)
        fw_log_std = torch.clamp(fw_log_std, LOG_STD_MIN, LOG_STD_MAX)
        return Normal(fw_mu, fw_log_std.exp())


class ProbabilisticSimpleForwardNetIncludingReward(nn.Module):
    def __init__(self, env, cfg):
        super().__init__()
        hidden_size = cfg["hidden_size"]
        if isinstance(env.observation_space, spaces.dict.Dict):
            self.obs_shape = np.sum(
                [obs_space.shape for obs_space in env.observation_space.spaces.values()]
            )
        else:
            self.obs_shape = np.prod(env.observation_space.shape)
        self.action_shape = int(np.prod(env.action_space.shape))

        self.state_action_encoder = nn.Sequential(
            nn.Linear(self.obs_shape + self.action_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        # obs_shape: int64
        # add one for reward
        self.fw_mu = nn.Linear(hidden_size, self.obs_shape + 1)
        self.fw_std = nn.Linear(hidden_size, self.obs_shape + 1)

    def forward(self, obs, action):
        # forward model: p(w' | w, a)
        hx = torch.cat([obs, action], dim=-1)
        hx = self.state_action_encoder(hx)
        hx = F.relu(hx)
        fw_mu, fw_log_std = self.fw_mu(hx), self.fw_std(hx)
        fw_log_std = torch.clamp(fw_log_std, LOG_STD_MIN, LOG_STD_MAX)
        return Normal(fw_mu, fw_log_std.exp())


class ProbabilisticForwardNetPositionPredictionIncludingReward(nn.Module):
    def __init__(self, env, cfg, maximum_number_of_objects: int = 5):
        super().__init__()
        hidden_size = cfg["hidden_size"]
        # 2 positions for the x, y coordinates of the agent
        # + 2 positions for the x, y coordinates of each object (maximum_number_of_objects)
        self.old_position_shape = 2 + 2 * maximum_number_of_objects
        self.action_shape = int(np.prod(env.action_space.shape))

        self.state_action_encoder = nn.Sequential(
            nn.Linear(self.old_position_shape + self.action_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        # obs_shape: int64
        # add one for reward
        self.fw_mu = nn.Linear(hidden_size, self.old_position_shape + 1)
        self.fw_std = nn.Linear(hidden_size, self.old_position_shape + 1)

    def forward(self, old_position, action):
        # foward model: p(w' | w, a)
        hx = torch.cat([old_position.to(device), action.to(device)], dim=-1)
        hx = self.state_action_encoder(hx)
        hx = F.relu(hx)
        fw_mu, fw_log_std = self.fw_mu(hx), self.fw_std(hx)
        fw_log_std = torch.clamp(fw_log_std, LOG_STD_MIN, LOG_STD_MAX)
        return Normal(fw_mu, fw_log_std.exp())


class ProbabilisticForwardNetPositionPrediction(nn.Module):
    def __init__(self, env, cfg, maximum_number_of_objects: int = 5):
        super().__init__()
        hidden_size = cfg["hidden_size"]
        # 2 positions for the x, y coordinates of the agent
        # + 2 positions for the x, y coordinates of each object (maximum_number_of_objects)
        self.old_position_shape = 2 + 2 * maximum_number_of_objects
        self.action_shape = int(np.prod(env.action_space.shape))

        self.state_action_encoder = nn.Sequential(
            nn.Linear(self.old_position_shape + self.action_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.fw_mu = nn.Linear(hidden_size, self.old_position_shape)
        self.fw_std = nn.Linear(hidden_size, self.old_position_shape)

    def forward(self, old_position, action):
        # foward model: p(w' | w, a)
        hx = torch.cat([old_position, action], dim=-1)
        hx = self.state_action_encoder(hx)
        hx = F.relu(hx)
        fw_mu, fw_log_std = self.fw_mu(hx), self.fw_std(hx)
        fw_log_std = torch.clamp(fw_log_std, LOG_STD_MIN, LOG_STD_MAX)
        return Normal(fw_mu, fw_log_std.exp())
