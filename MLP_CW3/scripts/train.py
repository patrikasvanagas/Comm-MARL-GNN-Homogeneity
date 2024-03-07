from MLP_CW3.utils.envs import make_env
from MLP_CW3.algorithms.algorithm import make_alg
from MLP_CW3.algorithms.utils.buffer import MultiAgentRolloutBuffer
from MLP_CW3.algorithms.utils.logger import PrintLogger, WandbLogger
import random

import hydra
import numpy as np
from collections import deque
from omegaconf import DictConfig
import torch

def train(
    total_num_env_steps,
    log_interval,
    agent_groups,
    save_interval,
    eval_interval,
    episodes_per_eval,
    env_instance,
    alg_instance,
    logger,
    cfg,
    **kwargs,
):
    envs = env_instance
    alg = alg_instance
    device = cfg.alg.model.device
    update_frequency = cfg.training.update_frequency

    marl_storage = MultiAgentRolloutBuffer(
        buffer_size=update_frequency,
        observation_spaces=envs.observation_space,
        action_spaces=envs.action_space,
        actors_hidden_dim=alg.actors_hidden_dims,
        critics_hidden_dim=alg.critics_hidden_dims,
        device=device,
        gae_lambda=alg.gae_lambda,
        gamma=cfg.alg.gamma,
        n_envs=cfg.env.parallel_envs
    )
    all_infos = deque(maxlen=10)

    # Initialize environemnt and storage
    obs = envs.reset()
    marl_storage.init_obs_hiddens(obs)

    num_steps = 0
    completed_episodes = 0
    last_log_t = 0
    while num_steps < total_num_env_steps:
        # Query storage for latest observation & hiddens 
        obs, _, actors_hiddens, critics_hiddens, _, _ = marl_storage.get_last()
        # Take action
        actions, actors_hiddens, critics_hiddens = alg.act(obs, actors_hiddens, critics_hiddens)
        
        # Query environment for new state
        obs, rewards, dones, infos = envs.step(actions.tolist())
        rewards = np.array(rewards).transpose(1, 0)
        marl_storage.add(obs, 
                        actions.transpose(1, 0).unsqueeze(-1), 
                        actors_hiddens, 
                        critics_hiddens,
                        rewards,
                        dones,
                        )
        # log episode data for completed episodes
        assert sum(dones) in [0, cfg.env.parallel_envs]; f"Right now only synchronous env termination is accepted {dones}"
        if all(dones):
            for i, info in enumerate(infos):
                    completed_episodes += 1
                    info["completed_episodes"] = completed_episodes
                    all_infos.append(info)
                    logger.log_episode(num_steps, info, agent_groups)
            # Reset environment and initialize storage
            obs = envs.reset()
            marl_storage.init_obs_hiddens(obs)
        
        num_steps += cfg.env.parallel_envs

        update_step = num_steps // cfg.env.parallel_envs
        # Agents update logic
        if update_step and update_step % update_frequency == 0:
            # Query all transition from storage
            batch_obs, batch_act, batch_actor_hiddens, batch_critic_hiddens, batch_rew, batch_done = marl_storage.get()
            # Update networks 
            loss_dict = alg.update(batch_obs, batch_act, batch_rew, batch_done, batch_actor_hiddens, batch_critic_hiddens)
            loss_dict["updates"] = update_step / update_frequency
            logger.log_metrics(loss_dict, "timestep", num_steps)
            # Reset storage after update
            marl_storage.reset()
            marl_storage.init_obs_hiddens(obs, actors_hiddens, critics_hiddens, dones)

        if num_steps >= last_log_t + log_interval and len(all_infos) > 1:
            logger.log_progress(all_infos, update_step, num_steps, total_num_env_steps, agent_groups)
            all_infos.clear()
            last_log_t = num_steps
        

@hydra.main(config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    # if cfg.training.logger == "wandb":
    print(cfg)
    logger = WandbLogger(
        team_name="mlpcw3",
        project_name="mpl_cw3_runs",
        mode="offline",
        cfg=cfg
    )
    # else:
    #     logger = PrintLogger(cfg)
    logger.training_mode()
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

    logger.info("Initialising environment")
    envs = make_env(cfg.seed, cfg.env)
    agent_groups = [
        [i for i in range(cfg.env.arguments.num_adversaries)],
        [cfg.env.arguments.num_adversaries + i for i in range(cfg.env.arguments.num_good)],
        ]
    
    logger.info("Initialising algorithm")
    alg = make_alg(cfg.alg, 
                   envs.observation_space,
                   envs.action_space,
                   agent_groups
                   )

    logger.info("Start training")
    train(total_num_env_steps=cfg.training.num_env_steps,
        log_interval=cfg.training.log_interval,
        agent_groups=agent_groups,
        save_interval=cfg.training.save_interval,
        eval_interval=cfg.training.eval_interval,
        episodes_per_eval=cfg.training.episodes_per_eval,
        env_instance=envs,
        alg_instance=alg,
        logger=logger,
        cfg=cfg 
)
if __name__ == "__main__":
    main()