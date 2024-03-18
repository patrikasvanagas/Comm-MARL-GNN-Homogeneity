from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from collections import defaultdict

import hydra
import numpy as np
import torch

def _squash_info(info):
    info = [i for i in info if i]
    new_info = {}
    keys = set([k for i in info for k in i.keys()])
    to_remove_keys = ["terminal_observation"]
    for k in keys:
        if "attention_maps" in k:
            to_remove_keys.append(k)
    for k in to_remove_keys:
        keys.remove(k)
    for key in keys:
        values = [d[key] for d in info if key in d]
        mean = np.mean(values, 0)
        new_info[key] = mean

    return new_info

def update_step_infos(infos, step_infos):
    for info, step_info in zip(infos, step_infos):
        for k, v in info.items():
            step_info[k].append(v)
    return step_infos


def evaluate(
    cfg,
    parallel_envs,
    eval_episodes,
    envs,
    alg,
    logger,
    total_steps,
    env_config,
    agent_groups
):
    device = cfg.alg.model.device

    if env_config.parallel_envs != parallel_envs:
        # change number of parallel environments
        env_config.parallel_envs = parallel_envs
        envs = hydra.utils.call(env_config, seed=cfg.seed)
    
    obs = envs.reset()
    actor_hiddens = [
        torch.zeros(parallel_envs, hidden_dim, device=device) for hidden_dim in alg.actors_hidden_dims
    ]
    critic_hiddens = [
        torch.zeros(parallel_envs, hidden_dim, device=device) for hidden_dim in alg.critics_hidden_dims
    ]
    all_infos = []
    all_step_infos = []
    all_task_embeddings = []
    step_infos = [defaultdict(list) for _ in range(cfg.env.parallel_envs)]
    while len(all_infos) < eval_episodes:
        obs = [torch.tensor(o, device=device) for o in obs]
        if not alg.recurrent:
            actor_hiddens = [
                torch.zeros(parallel_envs, hidden_dim, device=device) for hidden_dim in alg.actors_hidden_dims
            ]
            critic_hiddens = [
                torch.zeros(parallel_envs, hidden_dim, device=device) for hidden_dim in alg.critics_hidden_dims
            ]
        with torch.no_grad():
            actions, actor_hiddens, critic_hiddens = alg.act(obs, actor_hiddens, critic_hiddens, evaluation=True)
        n_obs, rew, done, infos = envs.step(actions.tolist())
        infos = alg.update_info(infos)
        step_infos = update_step_infos(infos, step_infos)
        obs = n_obs
        for i, (info, d) in enumerate(zip(infos, done)):
            if d:
                all_infos.append(info)
                all_step_infos.append(step_infos[i])
                if alg.recurrent:
                    for actor_hidden, critic_hidden in zip(actor_hiddens, critic_hiddens):
                        actor_hidden[i, :].zero_()
                        critic_hidden[i, :].zero_()
                step_infos[i] = defaultdict(list)
    for group_id in range(len(agent_groups)):
        if group_id == 0:
            if f"attention_maps_{group_id}" in all_step_infos[0]:
                logger.log_heatmap(total_steps, np.mean(all_step_infos[0][f"attention_maps_{group_id}"], axis=0), main_label="Eval")
    eval_info = _squash_info(all_infos)
    step_infos = _squash_info(all_step_infos)
    if logger:
        logger.log_episode(total_steps, eval_info, step_infos, agent_groups,main_label="Eval")
    return eval_info, all_task_embeddings

