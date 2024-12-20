import os
from gym.spaces import Box

import numpy as np
import torch
from MLP_CW3.algorithms.a2c import A2C
from MLP_CW3.algorithms.encoders import Encoder
from gym.spaces.utils import flatdim

class A2CEncoder(A2C):
    def __init__(
        self,
        observation_space,
        action_space,
        agent_groups,
        cfg
    ):
        self._obs_sizes = np.array([flatdim(obs_space) for obs_space in observation_space])
        encoder_space = [Box(-np.inf, np.inf, (cfg.model.encoder_dim, )) for _ in observation_space]
        super(A2CEncoder, self).__init__(encoder_space, action_space, agent_groups, cfg)
        self.groups_encoder = [Encoder(self._obs_sizes[agent_group[0]], cfg.model.encoder_dim) for agent_group in self.agent_groups]

        self.groups_optimisers = [
                torch.optim.Adam(list(actors.parameters()) + list(critics.parameters()) + list(encoder.parameters()), self.lr) 
                for actors, critics, encoder in zip(self.groups_actors, self.groups_critics, self.groups_encoder)
            ]
        
        self.group_saveables = [
            {
                "actors": actors.state_dict(),
                "critics": critics.state_dict(),
                "encoder": encoder.state_dict(),
                "optimiser": optim.state_dict(),
            } for actors, critics, encoder, optim in zip(self.groups_actors, self.groups_critics, self.groups_optimisers, self.groups_encoder)
        ]
        print("Agent encoders:")
        for group_id, encoder in enumerate(self.groups_encoder):
            print(f"------------------")
            print(f"Group = {group_id}")
            print(encoder)

        
    def save(self, save_dir, episode):
        ep_save_dir = os.path.join(save_dir, f"e_{episode}")
        os.makedirs(ep_save_dir, exist_ok=True)

        for agent_group, group_saveable in zip(self.agent_groups, self.group_saveables):
            agent_group_str = str(agent_group)[1:-1].replace(' ', '').replace(',', '')
            model_name = f"model_group{agent_group_str}_e{episode}.pt"
            model_path = os.path.join(ep_save_dir, model_name)
            torch.save(group_saveable, model_path)
        return ep_save_dir

    def restore(self, path):
        # find mapping from agent group names to model file in path
        model_per_group = {}
        assert os.path.isdir(path)
        for f in os.listdir(path):
            group_ids = f.split("_")[1][5:]
            model_path = os.path.join(path, f)
            model_per_group[group_ids] = model_path

        for agent_group, group_actors, group_critics, group_optim, group_encoder in zip(self.agent_groups, self.groups_actors, self.groups_critics, self.groups_optimisers, self.groups_encoder):
            group_ids = str(agent_group).lstrip("[").rstrip("]").replace(",", "").replace(" ", "")
            model_path = model_per_group[group_ids]
            checkpoint = torch.load(model_path, map_location=torch.device(self.device))
            if isinstance(checkpoint["optimiser"], dict):
                group_actors.load_state_dict(checkpoint["actors"])
                group_critics.load_state_dict(checkpoint["critics"])
                group_encoder.load_state_dict(checkpoint["encoder"])
                group_optim.load_state_dict(checkpoint["optimiser"])
            else:
                group_actors.load_state_dict(checkpoint["actors"].state_dict())
                group_critics.load_state_dict(checkpoint["critics"].state_dict())
                group_encoder.load_state_dict(checkpoint["encoder"].state_dict())
                group_optim.load_state_dict(checkpoint["optimiser"].state_dict())

    def _query_actors(self, obss, hiddens, group_id):
        group_encoder = self.groups_encoder[group_id]
        
        agent_group = self.agent_groups[group_id]
        group_actors = self.groups_actors[group_id]
        group_obss = [obss[i] for i in agent_group]
            # Encoder Observations
        group_obss = group_encoder(torch.stack(group_obss))
        return group_actors(group_obss, hiddens[agent_group])

    def _query_critics(self, obss, hiddens, group_id, evaluation=False):
        if evaluation:
            values = [None for _ in range(self.n_agents)]
            new_hiddens = [None for _ in range(self.n_agents)]
        else:
            group_encoder = self.groups_encoder[group_id]
            agent_group = self.agent_groups[group_id]
            group_critics = self.groups_critics[group_id]
            group_obss = [obss[i] for i in agent_group]
            # Encoder Observations
            group_obss = group_encoder(torch.stack(group_obss))
            values, new_hiddens = group_critics(group_obss, hiddens[agent_group])
        return values, new_hiddens

    def _query_target_critics(self, obss, hiddens, group_id, evaluation=False):
        if evaluation:
            values = [None for _ in range(self.n_agents)]
            new_hiddens = [None for _ in range(self.n_agents)]
        else:
            group_encoder = self.groups_encoder[group_id]
            agent_group = self.agent_groups[group_id]
            group_critics = self.groups_tar_critics[group_id]
            group_obss = [obss[i] for i in agent_group]
            # Encoder Observations
            group_obss = group_encoder(torch.stack(group_obss))
            values, new_hiddens = group_critics(group_obss, hiddens[agent_group])
        return values, new_hiddens
