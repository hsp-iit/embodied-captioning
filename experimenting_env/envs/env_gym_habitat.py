import importlib
import logging
from typing import TYPE_CHECKING, Optional, Type

import gym
import habitat
import habitat_sim
import numpy as np
import torch
from habitat import Dataset
from habitat.core.environments import RLTaskEnv
from habitat.gym.gym_wrapper import HabGymWrapper

from experimenting_env.utils import projection_utils as pu

if TYPE_CHECKING:
    from omegaconf import DictConfig

from habitat.sims.habitat_simulator.actions import HabitatSimActions


class DisagreementUtil:
    def __init__(self):
        self.class_features = {}
        self.model = None
        self.pcd = None
        self.disagreement_map = None
        self.last_reward = 0.0
        self._episode_sum_reward = 0.0
        self.object_counter = 1000
        self.map_scale = 0.025

    def reset(self):
        self.pcd = None
        self.disagreement_map = None

    def _update_pointcloud(self, observations, bounds, episode):
        instances = observations['bbs']
        preds = instances['instances'].to("cpu")

        pose = observations['position']
        infos = [
            {'episode': episode, 'id_object': self.object_counter + i}
            for i in range(len(preds))
        ]
        self.object_counter += len(preds)

        point_cloud = pu.project_semantic_masks_to_3d(
            observations['depth'].squeeze(),
            pose,
            preds,
            infos,
        )
        point_cloud._episode = episode

        lower_bound, upper_bound = bounds
        logging.debug(f'Bounds {bounds}')
        if self.pcd is None or self.pcd._episode != point_cloud._episode:
            self.pcd: pu.SemanticPointCloud = point_cloud

        else:
            point_cloud.preprocess(lower_bound, upper_bound)

            self.pcd += point_cloud

        if len(preds) == 0:
            return

        self.pcd.preprocess(lower_bound, upper_bound)

    def get_distance(self, observations):
        lower_bound, upper_bound = observations['position']['bounds']
        maps = self.pcd.get_topdown_semantic(lower_bound, upper_bound, self.map_scale)[
            :, :, -2
        ]

        class_to_find = observations['objectgoal']

        # sum 1 to class_to_find
        possible_goals = torch.tensor(maps == class_to_find + 1).nonzero(as_tuple=True)

        current_pos = observations['position']['position']
        grid_x = (current_pos[0] / self.map_scale).astype(int) - int(
            np.floor(lower_bound[0] / self.map_scale)
        )  # columns in numpy
        grid_y = (current_pos[2] / self.map_scale).astype(int) - int(
            np.ceil(lower_bound[2] / self.map_scale)  # corresponding to z values
        )  # rows in numpy
        grid_xy = torch.tensor([grid_y, grid_x])
        grid_pos = torch.tensor([current_pos[2], current_pos[0]])  # yx in real coords
        if len(possible_goals[0]) == 0:
            return torch.tensor(10.0)
        else:

            goals_in_env = (
                (possible_goals[0] + int(np.ceil(lower_bound[2] / self.map_scale)))
                * self.map_scale,
                (possible_goals[1] + int(np.ceil(lower_bound[0] / self.map_scale)))
                * self.map_scale,
            )  # yx in real coords

            distances = torch.linalg.norm(
                grid_pos[:, None].repeat(1, len(possible_goals[0])).float()
                - torch.stack(goals_in_env),
                axis=0,
            )

            return distances.min()

    def get_maps(self, bounds, infos):
        lower_bound, upper_bound = bounds

        maps = self.pcd.get_topdown_semantic(
            lower_bound, upper_bound, map_scale=self.map_scale
        )
        return maps


@habitat.registry.register_env(name="GymHabitatEnv-v2")
class GymHabitatEnv(gym.Wrapper):
    """
    A registered environment that wraps a RLTaskEnv with the HabGymWrapper
    to use the default gym API.
    """

    def __init__(self, config: "DictConfig", dataset: Optional[Dataset] = None):
        base_env = RLTaskEnv(config=config, dataset=dataset)
        env = HabGymWrapper(base_env)
        self.current_goal = None
        self._current_scene = None
        self._disagreement_util = DisagreementUtil()
        super().__init__(env)

    def step(self, action):
        out = super().step(action)
        return out

    def update_pointcloud(self, obs, bounds, episode):
        self._disagreement_util._update_pointcloud(obs, bounds, episode)

    def get_distance(self, obs):

        return self._disagreement_util.get_distance(obs)

    def get_maps(self, bounds, infos):
        return self._disagreement_util.get_maps(bounds, infos)

    def _build_follower(self):
        if self._current_scene != self.env._env._env._sim.habitat_config.scene:

            self._follower = self.env._env._env._sim.make_greedy_follower(
                0,
                stop_key=HabitatSimActions.stop,
                forward_key=HabitatSimActions.move_forward,
                left_key=HabitatSimActions.turn_left,
                right_key=HabitatSimActions.turn_right,
            )
            self._current_scene = self.env._env._env._sim.habitat_config.scene

    def get_action_to_goal(self):
        goal_reached = False

        self._build_follower()
        if self.current_goal is None:
            act = 2 # turn let
        else:
            
            try:
                act = self._follower.next_action_along(
                    np.array(self.current_goal[0].position)
                )
            except Exception as ex:
                act = 0
            if act == 0:
                goal_reached = True

        return act, goal_reached

    def set_goals(self, data):
        self.current_goal = data
