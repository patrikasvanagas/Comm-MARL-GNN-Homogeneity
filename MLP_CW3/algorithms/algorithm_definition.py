from abc import ABC

from gym.spaces.utils import flatdim
import numpy as np
import collections

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class Algorithm(ABC):
    def __init__(
        self,
        observation_spaces,
        action_spaces,
        algorithm_config,
    ):
        self.n_agents = len(observation_spaces)
        self.obs_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.config = algorithm_config

        self.obs_sizes = np.array([flatdim(obs_space) for obs_space in observation_spaces])
        self.obs_shapes = np.array([obs_space.shape for obs_space in observation_spaces])
        self.action_sizes = np.array([flatdim(act_space) for act_space in action_spaces])
        self.act_nums = np.array([act_space.n for act_space in action_spaces])

        # set all values from config as attributes
        for k, v in flatten(algorithm_config).items():
            setattr(self, k, v)