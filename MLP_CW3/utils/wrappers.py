from collections import deque
import itertools
import random
from time import perf_counter
from typing import Iterable

import gym
from gym import ObservationWrapper, spaces
import numpy as np
import torch


class RecordEpisodeStatistics(gym.Wrapper):
    """ Multi-agent version of RecordEpisodeStatistics gym wrapper"""

    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.t0 = perf_counter()
        self.episode_reward = np.zeros(self.n_agents)
        self.episode_length = 0
        self.reward_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        self.episode_reward = np.zeros(self.n_agents)
        self.episode_length = 0
        self.t0 = perf_counter()

        return observation

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self.episode_reward += np.array(reward, dtype=np.float64)
        self.episode_length += 1
        if all(done):
            info["episode_reward"] = self.episode_reward.copy()
            for i, agent_reward in enumerate(self.episode_reward):
                info[f"agent{i}/episode_reward"] = agent_reward
            info["episode_length"] = self.episode_length
            info["episode_time"] = perf_counter() - self.t0

            self.reward_queue.append(self.episode_reward.copy())
            self.length_queue.append(self.episode_length)
        return observation, reward, done, info


class AgentOneHotIdWrapper(gym.Wrapper):
    """
    Wrapper that adds OneHot Agent IDs to their observations
    """

    def __init__(self, env):
        super().__init__(env)
        obs_spaces = []
        self.n_agents = env.n_agents
        for space in self.observation_space:
            shape = list(space.shape)
            shape[0] += self.n_agents
            shape = tuple(shape)
            obs_spaces.append(
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=shape,
                    dtype=np.float32,
                )
            )
        self.observation_space = spaces.Tuple(tuple(obs_spaces))

    def insert_agent_id(self, observation):
        observation_with_ids = []
        for agent_id, agent_obs in enumerate(observation):
            observation_with_ids.append(np.insert(agent_obs, 0, np.eye(self.n_agents)[agent_id]))
        return tuple(observation_with_ids)

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        observation_with_ids = self.insert_agent_id(observation)
        return observation_with_ids

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation_with_ids = self.insert_agent_id(observation)
        return observation_with_ids, reward, done, info


class ConcatDictObservation(ObservationWrapper):
    """
    Wrapper which concatenates flattened image and feature vector of agents
    """
    def __init__(self, env):
        super(ConcatDictObservation, self).__init__(env)

        assert all([isinstance(obs_space, spaces.Dict) for obs_space in self.observation_space])
        assert all(["image" in obs_space.spaces and "features" in obs_space.spaces for obs_space in self.observation_space])
        self.image_spaces = [obs_space["image"] for obs_space in self.observation_space]

        ma_spaces = []
        for sa_obs in env.observation_space:
            flat_image_dim = spaces.flatdim(sa_obs["image"])
            features_dim = spaces.flatdim(sa_obs["features"])
            flatdim = flat_image_dim + features_dim

            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple(
            [
                np.concatenate(
                    [
                        spaces.flatten(image_space, obs["image"]),
                        obs["features"],
                    ]
                )
                for image_space, obs in zip(self.image_spaces, observation)
            ]
        )


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple(
            [
                spaces.flatten(obs_space, obs)
                for obs_space, obs in zip(self.env.observation_space, observation)
            ]
        )


class SquashDones(gym.Wrapper):
    r"""Wrapper that squashes multiple dones to a single one using all(dones)"""

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, all(done), info


class GlobalizeReward(gym.RewardWrapper):
    def reward(self, reward):
        return self.n_agents * [sum(reward)]


class StandardizeReward(gym.RewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stdr_wrp_sumw = np.zeros(self.n_agents, dtype=np.float32)
        self.stdr_wrp_wmean = np.zeros(self.n_agents, dtype=np.float32)
        self.stdr_wrp_t = np.zeros(self.n_agents, dtype=np.float32)
        self.stdr_wrp_n = 0
        
    def reward(self, reward):
        # based on http://www.nowozin.net/sebastian/blog/streaming-mean-and-variance-computation.html
        # update running avg and std
        weight = 1.0

        q = reward - self.stdr_wrp_wmean
        temp_sumw = self.stdr_wrp_sumw + weight
        r = q * weight / temp_sumw
        
        self.stdr_wrp_wmean += r
        self.stdr_wrp_t += q*r*self.stdr_wrp_sumw
        self.stdr_wrp_sumw = temp_sumw
        self.stdr_wrp_n += 1

        if self.stdr_wrp_n == 1:
            return reward

        # calculate standardized reward
        var = (self.stdr_wrp_t * self.stdr_wrp_n) / (self.stdr_wrp_sumw*(self.stdr_wrp_n-1))
        stdr_rew = (reward - self.stdr_wrp_wmean) / (np.sqrt(var) + 1e-6)
        return stdr_rew


class TimeLimit(gym.wrappers.TimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not all(done)
            done = len(observation) * [True]
        return observation, reward, done, info
