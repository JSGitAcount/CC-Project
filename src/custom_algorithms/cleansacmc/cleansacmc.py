from typing import Dict, Optional, Tuple, Union

import pathlib
import io
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from gymnasium import spaces

from stable_baselines3.common.logger import Logger
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from .mc import MorphologicalNetworks

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(nn.Module):
    def __init__(self, env, hidden_size):
        super().__init__()
        if isinstance(env.observation_space, spaces.dict.Dict):
            obs_shape = np.sum(
                [obs_space.shape for obs_space in env.observation_space.spaces.values()]
            )
        else:
            obs_shape = np.sum(env.observation_space.shape)
        self.fc1 = nn.Linear(obs_shape, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(hidden_size, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class Critic(nn.Module):
    def __init__(self, env, hidden_size):
        super().__init__()
        if isinstance(env.observation_space, spaces.dict.Dict):
            obs_shape = np.sum(
                [obs_space.shape for obs_space in env.observation_space.spaces.values()]
            )
        else:
            obs_shape = np.sum(env.observation_space.shape)
        self.fc1 = nn.Linear(obs_shape + np.prod(env.action_space.shape), hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class CriticEnsemble(nn.Module):
    def __init__(self, env, n_critics: int, hidden_size):
        super().__init__()
        self._critics = nn.ModuleList(
            [Critic(env, hidden_size) for _ in range(n_critics)]
        )

    def forward(self, x, a):
        return torch.stack([critic(x, a) for critic in self._critics])


def flatten_obs(obs, device):
    if not isinstance(obs, dict):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs)
        return obs.to(device)
    observation, ag, dg = obs["observation"], obs["achieved_goal"], obs["desired_goal"]
    if isinstance(observation, np.ndarray):
        observation = torch.from_numpy(observation).to(device)
    if isinstance(ag, np.ndarray):
        ag = torch.from_numpy(ag).to(device)
    if isinstance(dg, np.ndarray):
        dg = torch.from_numpy(dg).to(device)
    return torch.cat([observation, ag, dg], dim=1).to(dtype=torch.float32)


class CLEANSACMC:
    """
    A one-file version of SAC derived from both the CleanRL and stable-baselines3 versions of SAC.
    :param env: The Gym environment to learn from
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Bctor and Value function)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically
    :param use_her: whether to use hindsight experience replay (HER) by using the SB3 HerReplayBuffer
    """

    def __init__(
            self,
            env: GymEnv,
            learning_rate: float = 3e-4,
            buffer_size: int = 1_000_000,
            learning_starts: int = 100,
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            ent_coef: Union[str, float] = "auto",
            use_her: bool = True,
            n_critics: int = 2,
            hidden_size: int = 256,
            mc: dict = {},
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.tau = tau
        self.gamma = gamma
        self.n_critics = n_critics
        self.hidden_size = hidden_size
        self.mc = mc

        self.env = env
        if isinstance(self.env.action_space, spaces.Box):
            assert np.all(
                np.isfinite(
                    np.array([self.env.action_space.low, self.env.action_space.high])
                )
            ), "Continuous action space must have a finite lower and upper bound"

        # initialize replay buffer
        if use_her:
            self.replay_buffer = HerReplayBuffer(
                self.buffer_size,
                self.env.observation_space,
                self.env.action_space,
                env=self.env,
                device=self.device,
                n_envs=self.env.num_envs,
            )
        elif isinstance(self.env.observation_space, spaces.dict.Dict):
            self.replay_buffer = DictReplayBuffer(
                self.buffer_size,
                self.env.observation_space,
                self.env.action_space,
                device=self.device,
                n_envs=self.env.num_envs
            )
        else:
            self.replay_buffer = ReplayBuffer(
                self.buffer_size,
                self.env.observation_space,
                self.env.action_space,
                device=self.device,
                n_envs=self.env.num_envs,
            )

        self._create_actor_critic()

        self.ent_coef = ent_coef
        if self.ent_coef == "auto":
            self.target_entropy = float(
                -np.prod(self.env.action_space.shape).astype(np.float32)
            )
            self.log_ent_coef = torch.zeros(1, device=self.device).requires_grad_(True)
            self.ent_coef_optimizer = torch.optim.Adam(
                [self.log_ent_coef], lr=self.learning_rate
            )
        else:
            self.ent_coef_tensor = torch.tensor(
                float(self.ent_coef), device=self.device
            )

        self.logger = None
        self._last_obs = None
        self.num_timesteps = 0
        self._n_updates = 0

    def _create_actor_critic(self) -> None:
        self.actor = Actor(self.env, self.hidden_size).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.learning_rate
        )
        self.critic = CriticEnsemble(self.env, self.n_critics, self.hidden_size).to(
            self.device
        )
        self.critic_target = CriticEnsemble(
            self.env, self.n_critics, self.hidden_size
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.learning_rate
        )

        self.mc_network = MorphologicalNetworks(self.env, self.mc).to(self.device)
        self.mc_optimizer = torch.optim.Adam(
            self.mc_network.parameters(), lr=self.mc["learning_rate"]
        )

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval=None,
    ):
        callback.init_callback(self)
        callback.on_training_start(locals(), globals())

        obs = self.env.reset()
        if len(obs) == 2:
            self._last_obs = obs[0]
        else:
            self._last_obs = obs

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollout(callback=callback)

            if continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                self.train()

        callback.on_training_end()

        return self

    def collect_rollout(self, callback: BaseCallback):
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :return: True if the training should continue, else False
        """
        # Select action randomly or according to policy
        if self.num_timesteps < self.learning_starts:
            actions, log_prob = np.array([self.env.action_space.sample()]), torch.zeros(1)
        else:
            actions, log_prob = self.predict(self._last_obs)

        # perform action
        new_obs, rewards, dones, infos = self.env.step(actions)
        self.calc_reward(
            flatten_obs(new_obs, self.device).float(),
            torch.from_numpy(actions).to(self.device).float(),
            torch.from_numpy(rewards).view(1, -1).to(self.device).float(),
            is_executing=True
        )
        self.logger.record("actor_entropy", -log_prob.mean().cpu().item(), exclude="tensorboard")
        self.logger.record("train/rollout_rewards_step", np.mean(rewards))
        self.logger.record_mean("train/rollout_rewards_mean", np.mean(rewards))
        self.num_timesteps += self.env.num_envs

        # save data to replay buffer; handle `terminal_observation`
        real_next_obs = new_obs.copy()
        for idx, done in enumerate(dones):
            if done:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        self.replay_buffer.add(
            self._last_obs, real_next_obs, actions, rewards, dones, infos
        )

        self._last_obs = new_obs

        # Only stop training if return value is False, not when it is None.
        if callback.on_step() is False:
            return False
        return True

    def train_mc(self, observations, next_observations, actions):
        # Generate a normal distribution over forward_normal and world_normal given observations and actions.
        forward_normal, world_normal = self.mc_network(observations, actions)
        # The loss is the negative log likelihood of the next obs being sampled from a distribution given by mu, sigma
        # Likelihood is a value between 0 and 1.
        # Therefore, the log likelihood is negative
        # So we negate the log likelihood it to obtain a value that we can minimize.
        #
        fw_loss = -forward_normal.log_prob(next_observations)
        w_prime_a_loss = -world_normal.log_prob(next_observations)
        loss = (fw_loss + w_prime_a_loss).mean()

        self.logger.record("mc/fw_loss", fw_loss.mean().item())
        self.logger.record("mc/w_prime_a_loss", w_prime_a_loss.mean().item())
        self.logger.record("mc/loss", loss.item())
        self.mc_optimizer.zero_grad()
        loss.backward()
        self.mc_optimizer.step()

    def calc_reward(self, observations, actions, e_rewards, is_executing=False):
        forward_normal, world_normal = self.mc_network(observations, actions)
        _err = (
            torch.distributions.kl.kl_divergence(forward_normal, world_normal)
            .mean(-1)
            .unsqueeze(1)
        )
        i_rewards = _err.clone() * self.mc["reward_eta"]
        rewards = e_rewards + i_rewards
        if is_executing:  # If this action is actually executed, log only the current step for visualizing the value.
            self.logger.record("mc/i_reward", i_rewards.mean().item())
            self.logger.record("mc/e_reward", e_rewards.mean().item())
            self.logger.record("mc/kld", _err.mean().item())
            self.logger.record("mc/reward", rewards.mean().item())
        else:  # Take mean of all logged values until dump.
            self.logger.record_mean("mc/i_reward", i_rewards.mean().item())
            self.logger.record_mean("mc/e_reward", e_rewards.mean().item())
            self.logger.record_mean("mc/kld", _err.mean().item())
            self.logger.record_mean("mc/reward", rewards.mean().item())
        return rewards

    def train(self):
        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

        # Sample replay buffer
        replay_data = self.replay_buffer.sample(self.batch_size)
        observations = flatten_obs(replay_data.observations, self.device)
        next_observations = flatten_obs(replay_data.next_observations, self.device)

        self.train_mc(observations, next_observations, replay_data.actions)

        # optimize entropy coefficient
        if self.ent_coef == "auto":
            with torch.no_grad():
                _, log_pi = self.actor.get_action(observations)
            ent_coef_loss = (-self.log_ent_coef * (log_pi + self.target_entropy)).mean()
            self.logger.record("train/ent_coef_loss", ent_coef_loss.item())

            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
            ent_coef = self.log_ent_coef.exp().item()
        else:
            ent_coef = self.ent_coef_tensor
        self.logger.record("train/ent_coef", ent_coef)

        # train critic
        with torch.no_grad():
            next_state_actions, next_state_log_pi = self.actor.get_action(
                next_observations
            )
            crit_next_targets = self.critic_target(
                next_observations, next_state_actions
            )
            min_crit_next_target = torch.min(crit_next_targets, dim=0).values
            min_crit_next_target -= ent_coef * next_state_log_pi

            rewards = self.calc_reward(
                observations, next_state_actions, replay_data.rewards
            )
            next_q_value = (
                    rewards.flatten()
                    + (1 - replay_data.dones.flatten())
                    * self.gamma
                    * min_crit_next_target.flatten()
            )

        critic_a_values = self.critic(observations, replay_data.actions)
        crit_loss = torch.stack(
            [F.mse_loss(_a_v, next_q_value.view(-1, 1)) for _a_v in critic_a_values]
        ).sum()
        self.logger.record("train/critic_loss", crit_loss.item())
        self.logger.record("train/train_rewards", replay_data.rewards.flatten().mean().item())

        self.critic_optimizer.zero_grad()
        crit_loss.backward()
        self.critic_optimizer.step()

        # train actor
        pi, log_pi = self.actor.get_action(observations)
        min_crit_pi = torch.min(self.critic(observations, pi), dim=0).values
        actor_loss = ((ent_coef * log_pi) - min_crit_pi).mean()
        self.logger.record("train/actor_loss", actor_loss.item())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks with polyak update
        for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.mul_(1 - self.tau)
            torch.add(
                target_param.data, param.data, alpha=self.tau, out=target_param.data
            )

    def predict(
            self,
            obs: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, None]:
        """
        Get the policy action given an observation.

        :param obs: the input observation
        :return: the model's action
        """
        observation = flatten_obs(obs, self.device)
        action, log_prob = self.actor.get_action(observation)
        return action.detach().cpu().numpy(), log_prob

    def save(self, path: Union[str, pathlib.Path, io.BufferedIOBase]):
        # Copy parameter list, so we don't mutate the original dict
        data = self.__dict__.copy()
        for to_exclude in [
            "logger",
            "env",
            "num_timesteps",
            "_n_updates",
            "_last_obs",
            "replay_buffer",
            "actor",
            "critic",
            "critic_target",
        ]:
            del data[to_exclude]
        # save network parameters
        data["_actor"] = self.actor.state_dict()
        data["_critic"] = self.critic.state_dict()
        data["_mc"] = self.mc_network.state_dict()
        torch.save(data, path)

    @classmethod
    def load(cls, path, env, **kwargs):
        model = cls(env=env, **kwargs)
        loaded_dict = torch.load(path)
        for k in loaded_dict:
            if k not in ["_actor", "_critic", "_mc"]:
                model.__dict__[k] = loaded_dict[k]
        # load network states
        model.actor.load_state_dict(loaded_dict["_actor"])
        model.critic.load_state_dict(loaded_dict["_critic"])
        model.mc_network.load_state_dict(loaded_dict["_mc"])
        return model

    def set_logger(self, logger: Logger) -> None:
        self.logger = logger

    def get_env(self):
        return self.env
