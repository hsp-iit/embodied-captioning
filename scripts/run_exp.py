import math
import multiprocessing
import os
from itertools import chain
import cv2
import habitat
import hydra
import magnum as mn
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import fog_of_war, maps
from habitat_baselines.agents.simple_agents import RandomAgent
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.core.environments import get_env_class
from habitat_sim.utils import common as sim_utils
import wandb
from experimenting_env.replay import SampleLoader
from experimenting_env.utils.sensors_utils import save_obs
from omegaconf import OmegaConf, open_dict


def run_exp(config, run_type: str, **kwargs) -> None:
    r"""This function runs the specified config with the specified runtype
    Args:
    config: Habitat.config
    runtype: str {train or eval}
    """

    trainer_init = baseline_registry.get_trainer(config.habitat_baselines.trainer_name)
    assert trainer_init is not None, f"{config.habitat_baselines.trainer_name} is not supported"
    trainer = trainer_init(config, **kwargs)

    if run_type == "train":
        trainer.train()
    elif run_type == "generate":
        trainer.generate()


@hydra.main(config_path='../confs/', config_name='train_policy.yaml')
def main(cfg) -> None:
    if not (os.path.exists(os.path.join(os.getcwd(), "data"))):
        os.symlink(cfg.data_base_dir, os.path.join(os.getcwd(), "data"))

    # TODO: understand how to load a habitat config
    hydra.core.global_hydra.GlobalHydra.instance().clear()    
    config = habitat.get_config(config_path=os.path.join(cfg.habitat_base_cfg_dir, cfg.habitat_cfg))

    run = wandb.init(
        mode='disabled',
        # Set the project where this run will be logged
        entity='tommaso-galliena01',
        project="SImCa",
        # Track hyperparameters and run metadata
        config=OmegaConf.to_container(config),
    )
    run_exp(config, config.ppo.mode if 'mode' in config.ppo else 'train', **cfg)


if __name__ == '__main__':
    main()
