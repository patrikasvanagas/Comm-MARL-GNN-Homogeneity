import os
from gym.spaces import Box

import numpy as np
import torch
from MLP_CW3.algorithms.a2c import A2C
from MLP_CW3.algorithms.encoders import (
    Encoder,
    GATNetwork,
    GATv2Network,
    GATv2Network_trainable_slope,
    AttentionMechanism_v2,
    AttentionMechanism_v3,
    CommMultiHeadAttention
)
from MLP_CW3.algorithms.similarity import similarity
from gym.spaces.utils import flatdim

GNN_MODELS_MAP = {
    "gat": GATNetwork,
    "gatv2": GATv2Network,
    "gatv2_trainable_slope": GATv2Network_trainable_slope,
    "attentionv2": AttentionMechanism_v2,
    "attentionv3": AttentionMechanism_v3,
    "mha": CommMultiHeadAttention
}

class A2CGNN(A2C):
    def __init__(self, observation_space, action_space, agent_groups, cfg):
        self._obs_sizes = np.array(
            [flatdim(obs_space) for obs_space in observation_space]
        )
        encoder_space = [
            Box(-np.inf, np.inf, (cfg.model.encoder_dim,)) for _ in observation_space
        ]
        super(A2CGNN, self).__init__(encoder_space, action_space, agent_groups, cfg)
        self.groups_encoder = [
            Encoder(self._obs_sizes[agent_group[0]], cfg.model.encoder_dim)
            for agent_group in self.agent_groups
        ]
        self.gnn_residual_connections = cfg.model.gnn_residual_connections
        self.groups_gnn = [GNN_MODELS_MAP[cfg.model.gnn_version](cfg.model.encoder_dim,
                                        cfg.model.encoder_dim,
                                        cfg.model.gnn_n_heads,
                                        cfg.model.gnn_use_masking,
                                        cfg.model.gnn_is_concat,
                                        cfg.model.gnn_dropout,
                                        cfg.model.gnn_leaky_relu_negative_slope,
                                        cfg.model.gnn_share_weights,
                                        ) 
                                        for _ in self.agent_groups]
        # self.groups_gnn = [
        #     AttentionMechanism(
        #         cfg.model.encoder_dim, cfg.model.encoder_dim, no_attention_to_self=False
        #     )
        #     for _ in self.agent_groups
        # ]

        self.groups_optimisers = [
            torch.optim.Adam(
                list(actors.parameters())
                + list(critics.parameters())
                + list(encoder.parameters())
                + list(gnn.parameters()),
                self.lr,
            )
            for actors, critics, encoder, gnn in zip(
                self.groups_actors,
                self.groups_critics,
                self.groups_encoder,
                self.groups_gnn,
            )
        ]
        self.group_saveables = [
            {
                "actors": actors.state_dict(),
                "critics": critics.state_dict(),
                "encoder": encoder.state_dict(),
                "gnn": gnn.state_dict(),
                "optimiser": optim.state_dict(),
            }
            for actors, critics, encoder, optim, gnn in zip(
                self.groups_actors,
                self.groups_critics,
                self.groups_encoder,
                self.groups_optimisers,
                self.groups_gnn,
            )
        ]
        self.device = cfg.model.device
        self.info_update_buffer = {}

        print("Agent encoders & GNNs:")
        for group_id, (encoder, gnn) in enumerate(
            zip(self.groups_encoder, self.groups_gnn)
        ):
            print(f"------------------")
            print(f"Group = {group_id}")
            print(encoder)
            print(gnn)

    def save(self, save_dir, step):
        ep_save_dir = os.path.join(save_dir, f"s_{step}")
        os.makedirs(ep_save_dir, exist_ok=True)

        for agent_group, group_saveable in zip(range(len(self.agent_groups)), self.group_saveables):
            agent_group_str = str(agent_group).replace(" ", "").replace(",", "")
            model_name = f"model_group{agent_group_str}_e{step}.pt"
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
        for (
            agent_group,
            group_actors,
            group_critics,
            group_optim,
            group_encoder,
            group_gnn,
        ) in zip(
            range(len(self.agent_groups)),
            self.groups_actors,
            self.groups_critics,
            self.groups_optimisers,
            self.groups_encoder,
            self.groups_gnn,
        ):
            group_ids = (
                str(agent_group)
                .lstrip("[")
                .rstrip("]")
                .replace(",", "")
                .replace(" ", "")
            )
            model_path = model_per_group[group_ids]
            checkpoint = torch.load(model_path, map_location=torch.device(self.device))
            if isinstance(checkpoint["optimiser"], dict):
                group_actors.load_state_dict(checkpoint["actors"])
                group_critics.load_state_dict(checkpoint["critics"])
                group_encoder.load_state_dict(checkpoint["encoder"])
                group_gnn.load_state_dict(checkpoint["gnn"])
                group_optim.load_state_dict(checkpoint["optimiser"])
            else:
                group_actors.load_state_dict(checkpoint["actors"].state_dict())
                group_critics.load_state_dict(checkpoint["critics"].state_dict())
                group_encoder.load_state_dict(checkpoint["encoder"].state_dict())
                group_gnn.load_state_dict(checkpoint["gnn"].state_dict())
                group_optim.load_state_dict(checkpoint["optimiser"].state_dict())

    def update_info(self, info):
        for group_id in range(len(self.agent_groups)):
            for i in range(len(info)):
                info[i][f"predator_similarity_{group_id}"] = self.info_update_buffer[f"predator_similarity_{group_id}"][i]
                info[i][f"encoder_similarity_{group_id}"] = self.info_update_buffer[f"encoder_similarity_{group_id}"][i]
                info[i][f"attention_maps_{group_id}"] = self.info_update_buffer[f"attention_maps_{group_id}"][i]

        return info
    
    def _query_actors(self, obss, hiddens, group_id):
        # obs - list of 1d tensor of observation for every agent, list, of 25-dim vectors



        group_encoder = self.groups_encoder[group_id]
        group_gnn = self.groups_gnn[group_id]

        agent_group = self.agent_groups[group_id]
        group_actors = self.groups_actors[group_id]
        # group_obs - a list of 1d tensors, each tensor is a vector of 128 elements.
        group_obss = [obss[i] for i in agent_group]
        # Encoder Observations
        # group_obss - a tensor of 
        group_obss = group_encoder(torch.stack(group_obss))
        self.info_update_buffer[f"encoder_similarity_{group_id}"] = similarity(group_obss.clone().detach().numpy())[0]
        # print(f"In query actors, group_obss shape: {group_obss.shape}. ")
        group_obss_gnn, attention_maps = group_gnn(group_obss)
        self.info_update_buffer[f"predator_similarity_{group_id}"] = similarity(group_obss.clone().detach().numpy())[0]
        self.info_update_buffer[f"attention_maps_{group_id}"] = attention_maps.clone().detach().numpy()[0]
        if self.gnn_residual_connections:
            group_obss_gnn += group_obss
        return group_actors(group_obss_gnn, hiddens[agent_group])

    def _query_critics(self, obss, hiddens, group_id, evaluation=False):
        if evaluation:
            values = [None for _ in range(self.n_agents)]
            new_hiddens = [None for _ in range(self.n_agents)]
        else:
            group_encoder = self.groups_encoder[group_id]
            group_gnn = self.groups_gnn[group_id]

            agent_group = self.agent_groups[group_id]
            group_critics = self.groups_critics[group_id]
            group_obss = [obss[i] for i in agent_group]
            # Encoder Observations
            group_obss = group_encoder(torch.stack(group_obss))
            group_obss_gnn, _ = group_gnn(group_obss)
            if self.gnn_residual_connections:
                group_obss_gnn += group_obss
            values, new_hiddens = group_critics(group_obss_gnn, hiddens[agent_group])
        return values, new_hiddens
    



    def _query_target_critics(self, obss, hiddens, group_id, evaluation=False):
        if evaluation:
            values = [None for _ in range(self.n_agents)]
            new_hiddens = [None for _ in range(self.n_agents)]
        else:
            group_encoder = self.groups_encoder[group_id]
            group_gnn = self.groups_gnn[group_id]

            agent_group = self.agent_groups[group_id]
            group_critics = self.groups_tar_critics[group_id]
            group_obss = [obss[i] for i in agent_group]
            # Encoder Observations
            group_obss = group_encoder(torch.stack(group_obss))
            group_obss_gnn, _ = group_gnn(group_obss)
            if self.gnn_residual_connections:
                group_obss_gnn += group_obss
            values, new_hiddens = group_critics(group_obss_gnn, hiddens[agent_group])
        return values, new_hiddens

