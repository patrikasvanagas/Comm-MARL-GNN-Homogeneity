import os
import random
import torch
import hydra
import time
from omegaconf import DictConfig
import supersuit as ss
import numpy as np

from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from stable_baselines3 import A2C

from pettingzoo.mpe import simple_tag_v3


def get_videos_path(model_name: str, write: bool = False):

    video_dir = "../../../videos"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    base_dir = f"{video_dir}/{model_name}"
    if write is True:
        return f"{base_dir}.mp4"
    return base_dir


def train(steps: int, env_cfg: DictConfig):
    env = simple_tag_v3.parallel_env(num_good=int(env_cfg.num_good),
                                     num_adversaries=int(env_cfg.num_adversaries),
                                     num_obstacles=int(env_cfg.num_obstacles),
                                     max_cycles=int(env_cfg.max_cycles),
                                     continuous_actions=bool(env_cfg.continuous_actions))

    # Pad the observations as the attackers have observation space (16), and the agent has observation space (14)
    env = ss.pad_observations_v0(env)
    env.reset()

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=steps)

    model_name = f"{env_cfg.name[0]}_{time.strftime('%Y%m%d-%H%M%S')}"
    file_path = get_videos_path(model_name)
    model.save(file_path)

    print(f"Model has been saved with name: {model_name}.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")
    env.close()

    return model_name


def visualise(model_name: str, seed: int, env_cfg: DictConfig):
    """
    Visualise the trained model by rendering the environment and saving the video to a file.
    The video is saved to the root directory of the project with the name "{model_name}.mp4".
    """
    model = A2C.load(get_videos_path(model_name))
    env = simple_tag_v3.env(render_mode="rgb_array",
                            num_good=int(env_cfg.num_good),
                            num_adversaries=int(env_cfg.num_adversaries),
                            num_obstacles=int(env_cfg.num_obstacles),
                            max_cycles=int(env_cfg.max_cycles),
                            continuous_actions=int(env_cfg.continuous_actions))
    env = ss.pad_observations_v0(env)
    env.reset(seed=seed)

    video_recorder = VideoRecorder(env, path=get_videos_path(model_name, write=True))

    for agent in env.agent_iter():
        """
        For each agent, take an action. The loop stops when the environment is terminated or truncated.
        """

        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            action = model.predict(observation, deterministic=True)[0]

        env.step(action)
        video_recorder.capture_frame()

    video_recorder.close()
    env.close()


@hydra.main(config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

    model_name = train(steps=cfg.training.num_env_steps, env_cfg=cfg.env)
    visualise(model_name=model_name, seed=cfg.seed, env_cfg=cfg.env)


if __name__ == "__main__":
    main()
