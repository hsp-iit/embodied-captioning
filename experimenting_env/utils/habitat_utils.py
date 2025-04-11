import math
import multiprocessing
import os
import random
from itertools import compress
from typing import List, TYPE_CHECKING

import numpy as np
import torch
from habitat import RLEnv, ThreadedVectorEnv, VectorEnv, make_dataset
from habitat.config import read_write
from habitat.gym import make_gym_from_config

if TYPE_CHECKING:
    from omegaconf import DictConfig


def make_env_fn(env_class, config, kwargs) -> RLEnv:
    r"""Creates an env of type env_class with specified config and rank.
    This is to be passed in as an argument when creating VectorEnv.
    Args:
        config: root exp config that has core env config node as well as
            env-specific config node.
        env_class: class type of the env to be created.
    Returns:
        env object created according to specification.
    """

    env = env_class(config=config, **kwargs)
    # env.seed(config.TASK_CONFIG.SEED)
    return env


def get_unique_scene_envs_generator(config, env_class, **kwargs):
    dataset = make_dataset(config.DATASET.TYPE, config=config.DATASET)

    scenes = config.DATASET.CONTENT_SCENES
    if "*" in config.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.DATASET)

    for i, scene_name in enumerate(scenes):
        task_config = config.clone()
        task_config.defrost()

        task_config.SEED = config.SEED + i
        task_config.DATASET.CONTENT_SCENES = [scene_name]

        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
            task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID
        )
        task_config.freeze()
        yield env_class(config=task_config, **kwargs)

def construct_envs(
    config: "DictConfig",
    workers_ignore_signals: bool = False,
    enforce_scenes_greater_eq_environments: bool = False,
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.

    :param config: configs that contain num_environments as well as information
    :param necessary to create individual environments.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor
    :param enforce_scenes_greater_eq_environments: Make sure that there are more (or equal)
        scenes than environments. This is needed for correct evaluation.

    :return: VectorEnv object created according to specification.
    """

    num_environments_per_gpu = config.habitat_baselines.num_environments_per_gpu
    num_environments_first_gpu = config.habitat_baselines.num_environments_first_gpu
    configs = []
    dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    scenes = config.habitat.dataset.content_scenes
    if "*" in config.habitat.dataset.content_scenes:
        scenes = dataset.get_scenes_to_load(config.habitat.dataset)
        semantic_filter_scenes = [
            os.path.exists(
                dataset.scene_ids[i].replace("//", "/").replace(".glb", "_semantic.ply")
            )
            for i in range(len(dataset.scene_ids))
        ]

        #scenes = list(compress(scenes, semantic_filter_scenes))

    num_devices = torch.cuda.device_count()
    num_environments = num_environments_first_gpu + num_environments_per_gpu * (torch.cuda.device_count() - 1)
    if num_environments < 1:
        raise RuntimeError("num_environments must be strictly positive")

    if len(scenes) == 0:
        raise RuntimeError(
            "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
        )

    random.shuffle(scenes)
    
    scene_splits: List[List[str]] = [[] for _ in range(num_environments)]
    if len(scenes) < num_environments:
        msg = f"There are less scenes ({len(scenes)}) than environments ({num_environments}). "
        print(f'+++++++++++++++++++++{len(scenes)}++++++++++++++++++++++++++++++')
        if enforce_scenes_greater_eq_environments:
            logger.warn(
                msg
                + "Reducing the number of environments to be the number of scenes."
            )
            num_environments = len(scenes)
            scene_splits = [[s] for s in scenes]
        else:
            logger.warn(
                msg
                + "Each environment will use all the scenes instead of using a subset."
            )
        for scene in scenes:
            for split in scene_splits:
                split.append(scene)
    else:
        for idx, scene in enumerate(scenes):
            scene_splits[idx % len(scene_splits)].append(scene)
        assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_environments):
        proc_config = config.copy()
        with read_write(proc_config):
            task_config = proc_config.habitat
            task_config.seed = task_config.seed + i
            if len(scenes) > 0:
                task_config.dataset.content_scenes = scene_splits[i]
            gpu_id = 0 if i < num_environments_first_gpu else (i - num_environments_first_gpu) % (num_devices -1) + 1
            proc_config['habitat']['simulator']['habitat_sim_v0']['gpu_device_id'] = gpu_id
        
        configs.append(proc_config)

    vector_env_cls: Type[Any]
    if int(os.environ.get("HABITAT_ENV_DEBUG", 0)):
        logger.warn(
            "Using the debug Vector environment interface. Expect slower performance."
        )
        vector_env_cls = ThreadedVectorEnv
    else:
        vector_env_cls = VectorEnv

    envs = vector_env_cls(
        make_env_fn=make_gym_from_config,
        env_fn_args=tuple((c,) for c in configs),
        workers_ignore_signals=workers_ignore_signals,
    )
    return envs

# def construct_envs(
#     config: "DictConfig",
#     workers_ignore_signals: bool = False,
#     enforce_scenes_greater_eq_environments: bool = False,
#     is_first_rank: bool = True,
#     **kwargs
# ) -> VectorEnv:
#     r"""Create VectorEnv object with specified config and env class type.
#     To allow better performance, dataset are split into small ones for
#     each individual env, grouped by scenes.
#     """
#
#     num_environments = config.habitat_baselines.num_environments
#     configs = []
#     dataset = make_dataset(config.habitat.dataset.type)
#     scenes = config.habitat.dataset.content_scenes
#     if "*" in config.habitat.dataset.content_scenes:
#         scenes = dataset.get_scenes_to_load(config.habitat.dataset)
#
#     if num_environments < 1:
#         raise RuntimeError("num_environments must be strictly positive")
#
#     if len(scenes) == 0:
#         raise RuntimeError(
#             "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
#         )
#
#     random.shuffle(scenes)
#
#     scene_splits: List[List[str]] = [[] for _ in range(num_environments)]
#     if len(scenes) < num_environments:
#         msg = f"There are less scenes ({len(scenes)}) than environments ({num_environments}). "
#         if enforce_scenes_greater_eq_environments:
#             # logger.warn(
#             #     msg
#             #     + "Reducing the number of environments to be the number of scenes."
#             # )
#             num_environments = len(scenes)
#             scene_splits = [[s] for s in scenes]
#         # else:
#             # logger.warn(
#             #     msg
#             #     + "Each environment will use all the scenes instead of using a subset."
#             # )
#         for scene in scenes:
#             for split in scene_splits:
#                 split.append(scene)
#     else:
#         for idx, scene in enumerate(scenes):
#             scene_splits[idx % len(scene_splits)].append(scene)
#         assert sum(map(len, scene_splits)) == len(scenes)
#
#     for env_index in range(num_environments):
#         proc_config = config.copy()
#         with read_write(proc_config):
#             task_config = proc_config.habitat
#             task_config.seed = task_config.seed + env_index
#             remove_measure_names = []
#             if not is_first_rank:
#                 # Filter out non rank0_measure from the task config if we are not on rank0.
#                 remove_measure_names.extend(
#                     task_config.task.rank0_measure_names
#                 )
#             if (env_index != 0) or not is_first_rank:
#                 # Filter out non-rank0_env0 measures from the task config if we
#                 # are not on rank0 env0.
#                 remove_measure_names.extend(
#                     task_config.task.rank0_env0_measure_names
#                 )
#
#             task_config.task.measurements = {
#                 k: v
#                 for k, v in task_config.task.measurements.items()
#                 if k not in remove_measure_names
#             }
#
#             if len(scenes) > 0:
#                 task_config.dataset.content_scenes = scene_splits[
#                     env_index
#                 ]
#
#         configs.append(proc_config)
#
#     vector_env_cls: Type[Any]
#     if int(os.environ.get("HABITAT_ENV_DEBUG", 0)):
#         # logger.warn(
#         #     "Using the debug Vector environment interface. Expect slower performance."
#         # )
#         vector_env_cls = ThreadedVectorEnv
#     else:
#         vector_env_cls = VectorEnv
#
#     envs = vector_env_cls(
#         make_env_fn=make_gym_from_config,
#         env_fn_args=tuple((c,) for c in configs),
#         workers_ignore_signals=workers_ignore_signals,
#     )
#
#     if config.habitat.simulator.renderer.enable_batch_renderer:
#         envs.initialize_batch_renderer(config)
#
#     return envs

# def construct_envs(
#     config: "DictConfig",
#     env_class: RLEnv,
#     workers_ignore_signals: bool = False,
#     mode="train",
#     **kwargs,
# ) -> VectorEnv:
#     r"""Create VectorEnv object with specified config and env class type.
#     To allow better performance, dataset are split into small ones for
#     each individual env, grouped by scenes.
#     :param config: configs that contain num_environments as well as information
#     :param necessary to create individual environments.
#     :param env_class: class type of the envs to be created.
#     :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor
#     :return: VectorEnv object created according to specification.
#     """
#
#     dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
#
#     scenes = config.habitat.dataset.content_scenes
#     if "*" in config.habitat.dataset.content_scenes:
#         scenes = dataset.get_scenes_to_load(config.habitat.dataset)
#         semantic_filter_scenes = [
#             os.path.exists(
#                 dataset.scene_ids[i].replace("//", "/").replace(".glb", "_semantic.ply")
#             )
#             for i in range(len(dataset.scene_ids))
#         ]
#
#         scenes = list(compress(scenes, semantic_filter_scenes))
#
#     (
#         sim_gpu_id,
#         num_processes,
#         num_processes_on_first_gpu,
#         num_processes_per_gpu,
#     ) = get_multi_gpu_config(len(scenes))
#
#     if mode != "train":
#         num_processes = min(num_processes, len(scenes))
#     configs = []
#
#     env_classes = [env_class for _ in range(num_processes)]
#
#     kwargs_per_env = [kwargs for _ in range(num_processes)]
#
#     random.shuffle(scenes)
#
#     scene_splits: List[List[str]] = [[] for _ in range(num_processes)]
#
#     for idx, scene in enumerate(scenes):
#         scene_splits[idx % len(scene_splits)].append(scene)
#
#     assert sum(map(len, scene_splits)) == len(scenes)
#     for i in range(num_processes):
#         task_config = config.clone()
#         task_config.defrost()
#
#         task_config.SEED = config.SEED + i
#         if len(scenes) > 0:
#             task_config.DATASET.CONTENT_SCENES = scene_splits[i]
#
#         if i < num_processes_on_first_gpu:
#             gpu_id = 0
#         else:
#             gpu_id = (i - num_processes_on_first_gpu) % (
#                 torch.cuda.device_count() - 1
#             ) + sim_gpu_id
#
#         task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id
#         task_config.SEED = random.randint(1, 10000)
#         task_config.freeze()
#         configs.append(task_config)
#
#     envs = ThreadedVectorEnv(
#         make_env_fn=make_env_fn,
#         env_fn_args=tuple(zip(env_classes, configs, kwargs_per_env)),
#         workers_ignore_signals=workers_ignore_signals,
#     )
#
#     return envs


def get_multi_gpu_config(num_scenes=25, x=10):
    # Automatically configure number of training threads based on
    # number of GPUs available and GPU memory size
    # total_num_scenes = num_scenes
    # gpu_memory = 100
    # num_gpus = torch.cuda.device_count()
    # for i in range(num_gpus):
    #     gpu_memory = min(
    #         gpu_memory,
    #         torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024,
    #     )
    #     if i == 0:
    #         assert (
    #             torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024
    #             > 10.0
    #         ), "Insufficient GPU memory"
    #
    # num_processes_per_gpu = int(gpu_memory / 1.4)
    #
    # num_processes_on_first_gpu = int((gpu_memory - x) / 1.4)
    #
    # sim_gpu_id = 0
    #
    # if num_gpus == 1:
    #     num_processes_on_first_gpu = num_processes_on_first_gpu
    #     num_processes_per_gpu = 0
    #     num_processes = num_processes_on_first_gpu
    # else:
    #     total_threads = (
    #         num_processes_per_gpu * (num_gpus - 1) + num_processes_on_first_gpu
    #     )
    #
    #     num_scenes_per_thread = math.ceil(total_num_scenes / total_threads)
    #     num_threads = math.ceil(total_num_scenes / num_scenes_per_thread)
    #     num_processes_per_gpu = min(
    #         num_processes_per_gpu, math.ceil(num_threads // (num_gpus - 1))
    #     )
    #
    #     num_processes_on_first_gpu = max(
    #         0, num_threads - num_processes_per_gpu * (num_gpus - 1)
    #     )
    #
    #     num_processes = num_processes_on_first_gpu + num_processes_per_gpu * (
    #         num_gpus - 1
    #     )  # num_threads
    #
    sim_gpu_id = 1

    num_processes = num_processes_on_first_gpu = 1
    num_processes_per_gpu = 0

    print("Auto GPU config:")
    print("Number of processes: {}".format(num_processes))
    print("Number of processes on GPU 0: {}".format(num_processes_on_first_gpu))
    print("Number of processes per GPU: {}".format(num_processes_per_gpu))
    return sim_gpu_id, num_processes, num_processes_on_first_gpu, num_processes_per_gpu
