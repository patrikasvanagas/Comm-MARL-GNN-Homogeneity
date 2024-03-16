from abc import ABC, abstractmethod
from collections import defaultdict
from hashlib import sha256
import json
import logging
import os
import shutil
import time
from typing import Dict

import numpy as np
from omegaconf import OmegaConf, DictConfig


logging.basicConfig(
    level=logging.INFO,
    format="(%(process)d) [%(levelname).1s] - (%(asctime)s) - %(name)s >> %(message)s",
    datefmt="%m/%d %H:%M:%S",
)


class NumpyQueue:
    """
    Class to hold queues computed as fixed-length numpy arrays for windows-averages
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.queues = defaultdict(lambda: np.zeros((capacity)))
        self.queues_length = defaultdict(int)

    def __len__(self):
        return self.length
    
    def add(self, key, value):
        self.queues[key][:-1] = self.queues[key][1:]
        self.queues[key][-1] = value
        self.queues_length[key] = min(self.queues_length[key] + 1, self.capacity)
    
    def mean(self, key):
        length = self.queues_length[key]
        if length == 0:
            return None
        entries = self.queues[key][-length:]
        assert len(entries) == length
        self.queues_length[key] = 0
        return entries.sum() / length
    
    def keys(self):
        return self.queues.keys()

class Logger(ABC):
    def __init__(self, cfg: DictConfig):
        self.config = OmegaConf.to_container(cfg)

        non_hash_keys = ["seed"]
        self.config_hash = sha256(
            json.dumps(
                {k: v for k, v in self.config.items() if k not in non_hash_keys},
                sort_keys=True,
            ).encode("utf8")
        ).hexdigest()[-10:]

        self.mode = "Training"
        self.wandb = None

        self.last_time = time.time()
        self.last_update = 0
        self.last_step = 0

    def info(self, *args, **kwargs):
        logging.info(*args, **kwargs)

    def debug(self, *args, **kwargs):
        logging.debug(*args, **kwargs)

    def warning(self, *args, **kwargs):
        logging.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        logging.error(*args, **kwargs)

    def critical(self, *args, **kwargs):
        logging.critical(*args, **kwargs)
    
    def watch(self, model):
        self.debug(model)
    
    def training_mode(self):
        self.mode = "Training"

    def finetuning_mode(self):
        self.mode = "Finetuning"
    
    @abstractmethod
    def _log_metrics(self, d: Dict, step_name: str, step: int):
        ...

    def log_metrics(self, d: Dict, step_name: str, step: int):
        d = {f"{self.mode}/{k}": v for k, v in d.items()}
        self._log_metrics(d, f"{self.mode}/{step_name}", step)
    
    def completed_run(self):
        pass

    def failed_run(self):
        pass
    
    def log_progress(
        self, infos, update, step, total_steps, groups,
    ):
        elapsed = time.time() - self.last_time
        self.last_time = time.time()
        ups = (update - self.last_update) / elapsed
        self.last_update = update
        steps_elapsed = step
        fps = (steps_elapsed - self.last_step) / elapsed
        self.last_step = step

        self.info(f"Updates {update}, Environment timesteps {steps_elapsed}")
        self.info(
            f"UPS: {ups:.1f}, FPS: {fps:.1f}, {steps_elapsed}/{total_steps} ({100 * steps_elapsed/total_steps:.2f}%) completed"
        )
        if infos:
            group_returns = ""
            for group_id, agent_group in enumerate(groups):
                mean_return = sum([sum([info[f"agent{agent_id}/episode_reward"].sum() for agent_id in agent_group]) / len(agent_group) for info in infos]) / len(infos) 
                group_returns += f" Group {group_id}: {mean_return:.3f}"

            self.info(f"Last {len(infos)} episodes with mean return:" + group_returns)

        self.info("-------------------------------------------")

    def log_episode(self, timestep, info, step_info, groups, main_label="Train", print_train_log=False):
        info["episode_reward"] = sum(info["episode_reward"])
        to_remove_keys = ["terminal_observation"]
        for k in info.keys():
            if "ground_rew" in k or "pos_rew" in k:
                to_remove_keys.append(k)
        for k in to_remove_keys:
            if k in info:
                del(info[k])

        log_dict = {}

        group_returns = ""

        if "predator_similarity" in step_info:
            log_dict[f"{main_label}/predator_similarity_mean"] = np.mean(step_info["predator_similarity"])
            log_dict[f"{main_label}/predator_similarity_std"] = np.std(step_info["predator_similarity"])

        for group_id, agent_group in enumerate(groups):
            mean_return = sum([info[f"agent{agent_id}/episode_reward"].sum() for agent_id in agent_group]) / len(agent_group)
            log_dict[f"{main_label}/group_{group_id}_mean_return"] = mean_return
            group_returns += f" Group {group_id}: {mean_return:.3f}"
        for k, v in info.items():
            log_dict[f"{main_label}/{k.replace('/','_')}"] = v
        self.log_metrics(log_dict, "timestep", timestep)
        if main_label == "Train":
            if print_train_log:
                self.info(
                    f"Completed episode {info['completed_episodes']}: Steps = {info['episode_length']} / Total Return = {info['episode_reward']:.3f}/ Total duration = {info['episode_time']}s" + group_returns 
                )
        else:
            self.info(
                f"Completed evaluation: Steps = {info['episode_length']} / Total Return = {info['episode_reward']:.3f} / Total duration = {info['episode_time']}s"+ group_returns
            )
    
       
class PrintLogger(Logger):
    def __init__(self, cfg):
        super(PrintLogger, self).__init__(cfg)
    
    def _log_metrics(self, d: Dict, step_name: str, step: int):
        self.info(f"---------- {step_name} = {step} -----------")
        for k, v in d.items():
            self.info(f"\t{k} = {v}")
        self.info("")

class WandbLogger(Logger):
    def __init__(self, team_name, project_name, mode, cfg):
        super().__init__(cfg)
        import wandb

        env_name = cfg.env.name if isinstance(cfg.env.name, str) else "-".join(cfg.env.name)
        alg_name = cfg.alg.name
        group_name = "_".join([env_name, alg_name, self.config_hash])

        self.wandb = wandb.init(
            entity=team_name,
            project=project_name,
            config=self.config,
            monitor_gym=True,
            group=group_name,
            mode=mode,
        )

        wandb_run_id = self.wandb.id
        self.info("*******************")
        self.info("WANDB RUN ID:")
        self.info(f"{wandb_run_id}")
        self.info("*******************")

        self.metrics_window = NumpyQueue(cfg.training.log_interval)
        self.step_name_by_metric = {}
    
    def _log_metrics(self, d: Dict, step_name: str, step: int):
        for key, v in d.items():
            if key not in self.step_name_by_metric:
                self.step_name_by_metric[key] = step_name
            self.metrics_window.add(key, v)
            self.metrics_window.add(f"{key}_T", step)

    def _push_data(self):
        # group metrics by steps
        metrics_by_step = {}
        for key in self.metrics_window.keys():
            if key.endswith("_T"):
                continue
            vt = self.metrics_window.mean(f"{key}_T")

            if vt is None:
                continue

            if vt in metrics_by_step:
                metrics_by_step[vt].append(key)
            else:
                metrics_by_step[vt] = [key]
            
        for steps, metric_keys in metrics_by_step.items():
            # add steps
            log_data = {self.step_name_by_metric[metric_keys[0]]: steps}
            # add metric values
            for key in metric_keys:
                log_data[key] = self.metrics_window.mean(key)
            self.wandb.log(log_data)
    
    def _save_file(self, path):
        wandb_save_dir = os.path.join(self.wandb.dir, path)
        shutil.copyfile(path, wandb_save_dir)

    def _save_dir(self, path):
        wandb_save_dir = os.path.join(self.wandb.dir, path)
        os.makedirs(wandb_save_dir, exist_ok=True)
        shutil.copytree(path, wandb_save_dir, dirs_exist_ok=True)

    def log_progress(self, infos, update, step, total_steps, groups):
        super().log_progress(infos, update, step, total_steps, groups)

        # save dirs and count stats if present
        for d in [d for d in os.listdir("./") if os.path.isdir(d)]:
            if d == "wandb" or d.startswith("."):
                continue
            self._save_dir(d)
        
        count_stats_path = os.path.join("./", "count_stats.csv")
        if os.path.isfile(count_stats_path):
            self._save_file(count_stats_path)

        self._push_data()
    
    def completed_run(self):
        self._push_data()

    def failed_run(self):
        self._push_data()
