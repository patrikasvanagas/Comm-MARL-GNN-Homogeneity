import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from MLP_CW3.algorithms.algorithm_definition import Algorithm
from MLP_CW3.algorithms.utils.models import MultiAgentNetworks, MultiAgentSharedNetworks
from MLP_CW3.algorithms.utils.utils import soft_update, hard_update
from torch.distributions.categorical import Categorical
import itertools


class A2C(Algorithm):
    def __init__(
        self,
        observation_space,
        action_space,
        agent_groups,
        cfg
    ):
        super(A2C, self).__init__(observation_space, action_space, cfg)
        self.agent_groups = agent_groups
        self.groups_actors = []
        self.groups_critics = []
        self.groups_tar_critics = []
        self.id_group_mapping = {} 

        for group_id, agent_group in enumerate(agent_groups):
            self.id_group_mapping = self.id_group_mapping | {agent_id:group_id for agent_id in agent_group}
            input_dims = self.obs_sizes[agent_group]
            output_dims = self.action_sizes[agent_group]
            critic_output_dims = [1] * len(agent_group)
            multi_agent_class = MultiAgentSharedNetworks if cfg.model.parameter_sharing else MultiAgentNetworks
            self.groups_actors.append(multi_agent_class(input_dims, output_dims, cfg.model.actor_hiddens, cfg.model.layernorm, cfg.model.device))
            self.groups_critics.append(multi_agent_class(input_dims, critic_output_dims, cfg.model.critic_hiddens, cfg.model.layernorm, cfg.model.device))
            self.groups_tar_critics.append(multi_agent_class(input_dims, critic_output_dims, cfg.model.critic_hiddens, cfg.model.layernorm, cfg.model.device))
        
        for tar_critics, critics  in zip(self.groups_tar_critics, self.groups_critics):
            hard_update(tar_critics, critics)

        self.groups_optimisers = [
                torch.optim.Adam(list(actors.parameters()) + list(critics.parameters()), self.lr) for actors, critics in zip(self.groups_actors, self.groups_critics)
            ]
        
        self.group_saveables = [
            {
                "actors": group_actors.state_dict(),
                "critics": group_critics.state_dict(),
                "optimiser": group_optim.state_dict(),
            } for group_actors, group_critics, group_optim in zip(self.groups_actors, self.groups_critics, self.groups_optimisers)
        ]

        print("Agent networks:")
        for group_id, (actors, critics, target_critics) in enumerate(zip(self.groups_actors, self.groups_critics, self.groups_tar_critics)):
            print(f"------------------")
            print(f"Group = {group_id}")
            print(actors)
            print(critics)
            print(target_critics)
    
    @property
    def actors_hidden_dims(self):
        return list(itertools.chain.from_iterable([actors.rnn_hidden_dims for actors in self.groups_actors]))

    @property
    def critics_hidden_dims(self):
        return list(itertools.chain.from_iterable([critics.rnn_hidden_dims for critics in self.groups_critics]))
        
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

        for agent_group, group_actors, group_critics, group_optim in zip(self.agent_groups, self.groups_actors, self.groups_critics, self.groups_optimisers):
            group_ids = str(agent_group).lstrip("[").rstrip("]").replace(",", "").replace(" ", "")
            model_path = model_per_group[group_ids]
            checkpoint = torch.load(model_path, map_location=torch.device(self.device))
            if isinstance(checkpoint["optimiser"], dict):
                group_actors.load_state_dict(checkpoint["actors"])
                group_critics.load_state_dict(checkpoint["critics"])
                group_optim.load_state_dict(checkpoint["optimiser"])
            else:
                group_actors.load_state_dict(checkpoint["actors"].state_dict())
                group_critics.load_state_dict(checkpoint["critics"].state_dict())
                group_optim.load_state_dict(checkpoint["optimiser"].state_dict())

    def _query_actors(self, obss, hiddens, group_id):
        agent_group = self.agent_groups[group_id]
        group_actors = self.groups_actors[group_id]
        group_obss = [obss[i] for i in agent_group]
        return group_actors(group_obss, hiddens[agent_group])

    def _query_critics(self, obss, hiddens, group_id, evaluation=False):
        if evaluation:
            values = [None for _ in range(self.n_agents)]
            new_hiddens = [None for _ in range(self.n_agents)]
        else:
            agent_group = self.agent_groups[group_id]
            group_critics = self.groups_critics[group_id]
            group_obss = [obss[i] for i in agent_group]
            values, new_hiddens = group_critics(group_obss, hiddens[agent_group])
        return values, new_hiddens

    def _query_target_critics(self, obss, hiddens, group_id, evaluation=False):
        if evaluation:
            values = [None for _ in range(self.n_agents)]
            new_hiddens = [None for _ in range(self.n_agents)]
        else:
            agent_group = self.agent_groups[group_id]
            group_critics = self.groups_tar_critics[group_id]
            group_obss = [obss[i] for i in agent_group]
            values, new_hiddens = group_critics(group_obss, hiddens[agent_group])
        return values, new_hiddens

    def act(self, obss, actor_hiddens, critic_hiddens=None, evaluation=False):
        """
        Choose action for agent given observation (always uses stochastic policy greedy)

        :param obss: observation of each agent (num_agents, parallel_envs, obs_space)
        :param task_embs: task embeddings of each agent (num_agents, parallel_envs, task_emb_dim)
        :param actor_hiddens: hidden states of each agent's actor (num_agents, parallel_envs, hidden_dim)
        :param critic_hiddens: hidden states of each agent's critic (num_agents, parallel_envs, hidden_dim)
        :param evaluation: boolean whether action selection is for evaluation
        :return: actions (num_agents, parallel_envs, 1), actor_hiddens (num_agents, parallel_envs, hidden_dim), critic_hiddens
        """
        actor_hiddens = torch.stack(actor_hiddens)
        critic_hiddens = torch.stack(critic_hiddens)

        actionss = []
        actor_hiddenss = []
        critic_hiddenss = []
        obss = [obs.unsqueeze(0) for obs in obss]
        actor_hiddens = actor_hiddens.unsqueeze(1)
        critic_hiddens = critic_hiddens.unsqueeze(1)

        with torch.no_grad():
            for group_id in range(len(self.agent_groups)):
                action_logits, act_hiddens = self._query_actors(obss, actor_hiddens, group_id)
                _, cri_hiddens = self._query_critics(obss, critic_hiddens, group_id)
                action_dists = [Categorical(logits=logits) for logits in action_logits]
                actions = [dist.mode() if (evaluation and self.greedy_evaluation) else dist.sample() for dist in action_dists]
                actions = [act.squeeze(0).unsqueeze(-1) for act in  actions]
                actionss.extend(actions)
                actor_hiddenss.extend(act_hiddens)
                critic_hiddenss.extend(cri_hiddens)

        # Prepare values for environment/storage
        actionss = torch.cat(actionss, dim=1)
        return actionss, actor_hiddenss, critic_hiddenss

    def _compute_returns(self, last_obs, last_hidden, rew, done_mask):
        """
        Compute n-step returns for all agents
        :param last_obs: batch of observations at last step for each agent (n_agents) x (parallel_envs, obs_shape)
        :param last_task_emb: batch of task embeddings at last step for each agent (n_agents) x (parallel_envs, task_emb_dim)
        :param last_hidden: batch of hidden states at last step for each agent (n_agents) x (parallel_envs, hidden_dim)
        :param rew: batch of rewards for each agent (n_agents) x (n_step, parallel_envs, 1)
        :param done_mask: batch of done masks (n_step + 1, parallel_envs, 1)
        """
        next_values_list = []
        last_obs = [torch.Tensor(o[-1]).unsqueeze(0) for o in last_obs]
        last_hidden = last_hidden.unsqueeze(1)
        with torch.no_grad():
            for group_id in range(len(self.agent_groups)):
                next_values, _ = self._query_target_critics(last_obs, last_hidden, group_id)
                next_values_list.extend(next_values.squeeze(1))
        n_step = done_mask.shape[1] - 1
        returns = [torch.stack(next_values_list)]
        for i in range(n_step - 1, -1, -1):
            ret = rew[:, i] + self.gamma * returns[0] * done_mask[:, i]
            returns.insert(0, ret)
        return torch.stack(returns[:-1], dim=1)


    def update(self, obs, act, rew, done_mask, actor_hiddens, critic_hiddens):
        """
        Compute and execute update
        :param obs: batch of observations for each agent (n_agents) x (n_step + 1, parallel_envs, obs_shape)
        :param act: batch of actions for each agent (n_agents) x (n_step, parallel_envs, 1)
        :param rew: batch of rewards for each agent (n_agents) x (n_step, parallel_envs, 1)
        :param done_mask: batch of done masks (joint for all agents) (n_step + 1, parallel_envs)
        :param task_embs: batch of task embeddings for each agent (n_agents) x (n_step + 1, parallel_envs, task_emb_dim)
        :param actor_hiddens: batch of hiddens for each agent's actor (n_agents) x (n_step + 1, parallel_envs, hidden_dim)
        :param critic_hiddens: batch of hiddens for each agent's critic (n_agents) x (n_step + 1, parallel_envs, hidden_dim)
        :return: dictionary of losses
        """
        rew = torch.Tensor(rew).unsqueeze(-1)
        done_mask = torch.Tensor(done_mask).unsqueeze(-1)
        act = torch.Tensor(act)
        actor_hiddens = torch.Tensor(actor_hiddens)
        critic_hiddens = torch.Tensor(critic_hiddens)

        # standardise rewards
        if self.standardise_rewards:
            rew = list(rew)
            for i in range(self.n_agents):
                rew[i] = (rew[i] - rew[i].mean()) / (rew[i].std() + 1e-5)

        last_critic_hiddens = critic_hiddens[:, -1, :]
        returns = self._compute_returns(obs, last_critic_hiddens, rew, done_mask)
        loss_dict = {}

        actor_hiddens = actor_hiddens[:, :-1, :]
        critic_hiddens = critic_hiddens[:, :-1, :]
        obss = [torch.Tensor(obs[:-1]) for obs in obs]

        for group_id, (agent_group, group_critics, target_critics, group_optim) in enumerate(zip(self.agent_groups, self.groups_critics, self.groups_tar_critics, self.groups_optimisers)):
            group_actions = act[agent_group]
            # Critics loss calculation for the group
            values, _ = self._query_critics(obss, actor_hiddens, group_id)
            advantages = returns[agent_group] - values
            value_loss = advantages.pow(2).mean()

            # Actors loss calculation for the group
            action_logits, _ = self._query_actors(obss, critic_hiddens, group_id)
            action_dists = [Categorical(logits=logits) for logits in action_logits]
            action_log_probs = torch.stack([dist.log_prob(a.squeeze()).unsqueeze(-1) for dist, a in zip(action_dists, group_actions)], dim=0)
            dist_entropy = torch.stack([dist.entropy().mean() for dist in action_dists], dim=0).mean()
            actor_loss = -(advantages.detach() * action_log_probs).mean()

            loss = actor_loss + value_loss - self.entropy_coef * dist_entropy
            loss_dict.update({
                f"Train/actor_loss_group{group_id}": actor_loss.item(),
                f"Train/value_loss_group{group_id}": value_loss.item(),
                f"Train/entropy_group{group_id}": dist_entropy.item(),
            })
            group_optim.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None and self.max_grad_norm != 0.0:
                nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
            group_optim.step()

            # update target networks
            soft_update(target_critics, group_critics, self.tau)

        return loss_dict
