import time

from MLP_CW3.utils.envs import make_env
import random

import hydra
import numpy as np
from omegaconf import DictConfig
import torch


@hydra.main(config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

    print(cfg.env)
    cfg.env["parallel_envs"] = 1
    env = make_env(cfg.seed, cfg.env)
    print(f"Reset observation: {env.reset()}")
    env.render("human")
    time.sleep(5000)
    actions = env.action_space.sample()
    observation, reward, done, info = env.step(actions)
    print(f"Step observation: {observation}")


if __name__ == "__main__":
    main()