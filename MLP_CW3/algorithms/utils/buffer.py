import numpy as np
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Generator, NamedTuple, Optional, Union, Tuple, List

import numpy as np
import torch
from gym import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.vec_env import VecNormalize

class RNNRolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    actor_hiddens: torch.Tensor
    critic_hiddens: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor

class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "cpu",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> RNNRolloutBufferSamples:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return torch.tensor(array).to(self.device)
        return torch.as_tensor(array).to(self.device)

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]], env: Optional[VecNormalize] = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        actor_hidden_dim: int,
        critic_hidden_dim: int,
        device: Union[torch.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.dones = None, None, None, None
        self.actor_hiddens, self.critic_hiddens = None, None
        self.generator_ready = False
        self.actor_hidden_dim = actor_hidden_dim
        self.critic_hidden_dim = critic_hidden_dim

        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size + 1, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size + 1, self.n_envs), dtype=np.float32)
        self.actor_hiddens = np.zeros((self.buffer_size + 1, self.n_envs, self.actor_hidden_dim), dtype=np.float32)
        self.critic_hiddens = np.zeros((self.buffer_size + 1, self.n_envs, self.critic_hidden_dim), dtype=np.float32)
        self.generator_ready = False
        super().reset()

    def add(
        self, obs: np.ndarray, action: np.ndarray, actor_hiddens: np.ndarray, critic_hiddens: np.ndarray, reward: np.ndarray, done: np.ndarray
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param done: End of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)
        # If done then clean the history of observations.
        masks = [done_==0 for done_ in done]
        self.observations[self.pos + 1] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.actor_hiddens[self.pos + 1] = np.array(actor_hiddens).copy()
        self.critic_hiddens[self.pos + 1] = np.array(critic_hiddens).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos + 1] = np.array(masks).copy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get_last(self) -> RNNRolloutBufferSamples:
        if not self.pos:
            return self._get_samples(0)
        return self._get_samples(self.pos - 1)
    
    def _get_samples(self, batch_inds: np.ndarray) -> RNNRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.actor_hiddens[batch_inds],
            self.critic_hiddens[batch_inds],
            self.rewards[batch_inds],
            self.dones[batch_inds],
        )
        return RNNRolloutBufferSamples(*tuple(map(self.to_torch, data)))

    def get(self):
        return self.observations, self.actions, self.actor_hiddens, self.critic_hiddens, self.rewards, self.dones


class MultiAgentRolloutBuffer:
    def __init__(
        self,
        buffer_size: int,
        observation_spaces: Tuple[spaces.Space],
        action_spaces: Tuple[spaces.Space],
        actors_hidden_dim: int,
        critics_hidden_dim: int,
        device: Union[torch.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.buffer_size = buffer_size
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.storages = [
            RolloutBuffer(buffer_size, 
                        observation_space,
                        action_space,
                        actor_hidden_dim,
                        critic_hidden_dim,
                        device,
                        gae_lambda,
                        gamma,
                        n_envs) 
                        for observation_space, action_space, actor_hidden_dim, critic_hidden_dim in 
                        zip(observation_spaces, action_spaces, actors_hidden_dim, critics_hidden_dim)]
    
    def reset(self) -> None:
        """
        Reset all storages
        """
        for storage in self.storages:
            storage.reset()

    def init_obs_hiddens(self, obs, actors_hidden=None, critics_hidden=None, dones = None):
        for agent_id, storage in enumerate(self.storages):
            step = storage.pos
            storage.observations[step] = obs[agent_id]

            # Initalize hiddens states with 0 if not provided
            if actors_hidden is not None:
                storage.actor_hiddens[step] = actors_hidden[agent_id]
            else:
                storage.actor_hiddens[step] =  np.zeros((storage.n_envs, storage.actor_hidden_dim), dtype=np.float32)
            
            if critics_hidden is not None:
                storage.critic_hiddens[step] = critics_hidden[agent_id]
            else:
                storage.critic_hiddens[step] = np.zeros((storage.n_envs, storage.actor_hidden_dim), dtype=np.float32)
        
            # Initalize dones states with 0 if not provided
            if dones is not None:
                storage.dones[step] = dones                
            else:
                storage.dones[step] = np.zeros((storage.n_envs), dtype=np.float32)

    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray) -> None:
        """
        Compute returns for all storages
        """
        for storage in self.storages:
            storage.compute_returns_and_advantage(last_values, dones)
    
    def add(self, obss, actions, actors_hiddens, critics_hiddens, rewards, dones) -> None:
        for agent_id, storage in enumerate(self.storages):
            storage.add(
                obss[agent_id],
                actions[agent_id],
                actors_hiddens[agent_id],
                critics_hiddens[agent_id],
                rewards[agent_id],
                dones,
            )

    def get_last(self) -> List[RNNRolloutBufferSamples]:
        return zip(*[storage.get_last() for storage in self.storages])

    def get(self):
        return zip(*[storage.get() for storage in self.storages])