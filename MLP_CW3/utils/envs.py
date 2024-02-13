import random
from functools import partial
from typing import Iterable

import gym
import numpy as np
from omegaconf import DictConfig
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import pettingzoo as pz

from MLP_CW3.utils import wrappers as _wrappers

def _make_parallel_envs(name, parallel_envs, dummy_vecenv, wrappers, seed, max_ep_length, arguments={}, fixed_arguments={}, argument_groups=None):
    def _env_thunk(env_seed):
        env = gym.make(name[0], **arguments)
        for wrapper in wrappers:
            wrap = getattr(_wrappers, wrapper) 
            if wrapper == "TimeLimit":
                assert max_ep_length is not None
                env = wrap(env, max_ep_length)
            env = wrap(env)
        env.seed(env_seed)
        return env

    if seed is None:
        seed = random.randint(0, 99999)

    env_thunks = [partial(_env_thunk, seed + i) for i in range(parallel_envs)]
    if dummy_vecenv:
        envs = DummyVecEnv(env_thunks)
        envs.buf_rews = np.zeros(
            (parallel_envs, len(envs.observation_space)), dtype=np.float32
        )
    else:
        envs = SubprocVecEnv(env_thunks, start_method="fork")
    return envs

def _make_env(name, wrappers, parallel_envs, dummy_vecenv, seed, max_ep_length, arguments={}, fixed_arguments={}, argument_groups=None):
    env = gym.make(name[0], **arguments)

    for wrapper in wrappers:
        wrap = getattr(_wrappers, wrapper) 
        if wrapper == "TimeLimit":
            assert max_ep_length is not None
            env = wrap(env, max_ep_length)
        env = wrap(env)

    env.seed(seed)
    return env


def make_env(seed, env):
    if env.parallel_envs:
        return _make_parallel_envs(**env, seed=seed)
    else:
        return _make_env(**env, seed=seed)