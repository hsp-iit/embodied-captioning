# type: ignore
import copy
import gc
import os
from collections import deque
from typing import Any, Dict, Union

import cv2
import hydra
import torch
import habitat
import habitat_sim
import magnum as mn
import numpy as np
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.nav import NavigationGoal, NavigationTask
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat_baselines.agents.simple_agents import GoalFollower, RandomAgent
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.core.environments import get_env_class
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_sim.utils import common as utils
from habitat.core.registry import registry
from habitat.tasks.utils import cartesian_to_polar
from experimenting_env.utils.astar2 import Grid, astar
from experimenting_env.utils.deprecated import *
from experimenting_env.utils.habitat_utils import (
    construct_envs,
    get_unique_scene_envs_generator,
)
from experimenting_env.utils.sensors_utils import save_obs
from experimenting_env.utils.skeleton import *
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    get_distrib_size,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)
from habitat.utils.geometry_utils import quaternion_rotate_vector


class Baseline(BaseRLTrainer):
    def __init__(self, config, exp_base_dir, agent_class, **kwargs):
        super().__init__(config)

        self.config = config
        self.kwargs = kwargs

        self.agent = agent_class(
            self.config.habitat_baselines.success_distance,
            self.config.habitat_baselines.goal_sensor_uuid,
        )

        self.exp_path = os.path.join(os.getcwd(), "dataset")

    def _init_train(self):
        envs = self._init_envs()
        os.makedirs(self.exp_path, exist_ok=True)

        self.current_observations = envs.reset()
        self.last_observations = []
        self.current_dones = []
        self.current_infos = []
        self.current_rewards = []
        return envs

    def _step(self, envs):
        actions = []
        for index_env in range(envs.num_envs):
            agent_position = envs.call_at(index_env, "get_agent_position")
            act = self.agent.act(self.current_observations[index_env], agent_position, index_env)
            actions.append(act)
            envs.async_step_at(index_env, act['action'])

        results = [envs.wait_step_at(index_env) for index_env in range(envs.num_envs)]

        self.last_observations = self.current_observations
        self.current_observations = [r[0] for r in results]
        self.last_actions = actions
        self.current_rewards = [r[1] for r in results]
        self.current_dones = [r[2] for r in results]
        self.current_infos = [r[3] for r in results]

        self.current_steps += 1

    def _init_envs(self, config=None, kwargs=None):
        if config is None:
            config = self.config
        if kwargs is None:
            kwargs = self.kwargs
        self.num_steps_done = 0
        envs = construct_envs(config, True)
        self.current_steps = np.zeros(envs.num_envs)
        return envs

    # def _init_envs(self, config=None, is_eval: bool = False):
    #     if config is None:
    #         config = self.config
    #     # if kwargs is None:
    #     #     kwargs = self.kwargs
    #     self.num_steps_done = 0
    #     # envs = construct_envs(
    #     #     config, get_env_class(config), True, **kwargs
    #     # )
    #     env_factory: VectorEnvFactory = hydra.utils.instantiate(
    #         config.habitat_baselines.vector_env_factory
    #     )
    #     envs = env_factory.construct_envs(
    #         config = config,
    #         workers_ignore_signals=is_slurm_batch_job(),
    #         enforce_scenes_greater_eq_environments=is_eval,
    #         is_first_rank=(
    #             not torch.distributed.is_initialized()
    #             or torch.distributed.get_rank() == 0
    #         ),
    #     )
    #     self.current_steps = np.zeros(envs.num_envs)
    #     return envs

    def train(self) -> None:
        pass

    def generate(self, config=None, kwargs=None) -> None:
        envs = self._init_train()

        generated_observations_paths = []
        self.exp_path = os.path.join(self.exp_path)
        while not self.is_done():
            self._step(envs)
            for idx in range(envs.num_envs):
                obs = self.current_observations[idx]

                done = self.current_dones[idx]
                episode = envs.current_episodes()[idx]

                paths = save_obs(
                    self.exp_path, episode.episode_id, obs, self.current_steps[idx]
                )

                generated_observations_paths.append(paths)
                self.num_steps_done += 1
                if done:
                    self.current_steps[idx] = 0
            if self.num_steps_done % 10 == 0:
                print(f"Progress: {int(self.percent_done() * 100)}%")
        del self.current_dones
        del self.current_observations

        envs.close()
        return sorted(generated_observations_paths)


@baseline_registry.register_trainer(name="randombaseline")
class RandomBaseline(Baseline):
    def __init__(self, config, exp_base_dir, **kwargs):
        super().__init__(config, exp_base_dir, RandomAgent, **kwargs)


# class ShortestPathAgent:
#     def __init__(self, env, goal_radius=0.1):
#         self.env = env
#         self.follower = ShortestPathFollower(env.habitat_env._sim, goal_radius, False)

#     def get_action(self):
#         best_action = self.follower.get_next_action(
#             self.env.current_episode.goals[0].position
#         )
#         return best_action


class BounceAgent(RandomAgent):
    def __init__(self, success_distance: float, goal_sensor_uuid: str) -> None:
        super().__init__(success_distance, goal_sensor_uuid)
        self.turn_count = 0

    def act(self, observations: Observations) -> Dict[str, Union[int, int]]:
        action = HabitatSimActions.MOVE_FORWARD

        # if collision with navmesh and not turning already
        if observations["agent_collision_sensor"] and self.turn_count == 0:
            self.turn_count = 16  # depends on TURN_ANGLE (where is it declared?)
            print("Collision:", observations["agent_collision_sensor"])

        if self.turn_count > 1:
            action = (
                HabitatSimActions.TURN_LEFT
            )  # TODO: choose turning side based on tangent angle wrt the obstacle

            self.turn_count -= 1
        elif self.turn_count == 1:
            action = HabitatSimActions.MOVE_FORWARD
            self.turn_count -= 1

        return {"action": action}


@baseline_registry.register_trainer(name="bouncebaseline")
class BounceBaseline(Baseline):
    def __init__(self, config, exp_base_dir, **kwargs):
        super().__init__(config, exp_base_dir, BounceAgent, **kwargs)


class RotateAgent(RandomAgent):
    def __init__(self, success_distance: float, goal_sensor_uuid: str) -> None:
        super().__init__(success_distance, goal_sensor_uuid)

    def act(self, observations: Observations) -> Dict[str, Union[int, int]]:
        return {"action": HabitatSimActions.TURN_LEFT}


@baseline_registry.register_trainer(name="rotatebaseline")
class RotateBaseline(Baseline):
    def __init__(self, config, exp_base_dir, **kwargs):
        super().__init__(config, exp_base_dir, RotateAgent, **kwargs)


@registry.register_task(name="FrontExp-v0")
class FrontierExplorationTask(NavigationTask):
    def __init__(self, config, sim, dataset):
        super().__init__(config=config, sim=sim, dataset=dataset)
        self.sim = sim

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return True


# @baseline_registry.register_trainer(name="randomgoalsbaseline")
# class RandomGoals(Baseline):
#     def __init__(self, config, exp_base_dir, **kwargs):
#         super().__init__(config, exp_base_dir, GoalFollower, **kwargs)
#         self.num_subgoals = 10

#     def compute_distance(self, a, b):
#         return np.linalg.norm(b - a)

#     def train(self):
#         pass

#     def _init_train(self):
#         envs = super()._init_train()

#         num_envs = envs.num_envs
#         self.counter = 0
#         self.visualize = True
#         self.counter = [0] * num_envs
#         self.sub_goals = [None] * num_envs
#         self.sub_goals_to_draw = [None] * num_envs
#         self.sub_goals_counter = [0] * num_envs
#         self.replan = [True] * num_envs
#         self.replan_retries = [0] * num_envs
#         return envs

#     def generate(self) -> None:
#         envs = self._init_train()

#         num_envs = envs.num_envs

#         goal_to_draw = [0, 0]
#         generated_observations_paths = []
#         while not self.is_done():
#             self._step(envs)

#             for idx in range(num_envs):
#                 self.counter[idx] += 1  # one counter per env?

#                 obs = self.current_observations[idx]
#                 info = self.current_infos[idx]
#                 done = self.current_dones[idx]
#                 n_step = envs.call_at(idx, "get_step")

#                 episode = envs.current_episodes()[idx]
#                 map_scale = 1
#                 generated_observations_paths.append(
#                     save_obs(self.exp_path, episode.episode_id, obs, n_step)
#                 )

#                 self.sub_goals_counter[idx] += 1

#                 if self.sub_goals_counter[idx] > self.num_subgoals:
#                     self.sub_goals_counter[idx] = 0
#                     if self.sub_goals[idx]:
#                         new_sub_goal = self.sub_goals[idx].pop(-1)
#                         if self.visualize:
#                             goal_to_draw = self.sub_goals_to_draw[idx].pop(-1)
#                         envs.call_at(
#                             idx,
#                             "set_goals",
#                             {
#                                 "data": [
#                                     NavigationGoal(
#                                         position=[new_sub_goal[0], 0, new_sub_goal[1]]
#                                     )
#                                 ]
#                             },
#                         )
#                     else:
#                         self.replan[idx] = True

#                 if (
#                     info is not None and obs is not None and self.replan[idx]
#                 ):  # (info['distance_to_goal'] < 0.5 or self.counter[idx] > 30): # time to replan
#                     # print("Replanning...")
#                     # self.counter[idx] = 0
#                     self.replan[idx] = False
#                     self.replan_retries[idx] = 0
#                     # allo_map = maps.get_top down_map_from_sim(self.sim, 1024, True, 0.05, 0)
#                     allo_map = info['top_down_map']['map']
#                     allo_map[allo_map > 1] = 1
#                     expl_map = info['top_down_map']['fog_of_war_mask']
#                     allo_map = cv2.resize(
#                         allo_map,
#                         (
#                             int(allo_map.shape[1] / map_scale),
#                             int(allo_map.shape[0] / map_scale),
#                         ),
#                     )
#                     expl_map = cv2.resize(
#                         expl_map,
#                         (
#                             int(expl_map.shape[1] / map_scale),
#                             int(expl_map.shape[0] / map_scale),
#                         ),
#                     )
#                     frontier_map = allo_map * 127 + expl_map * 127

#                     # compute frontiers
#                     edges = cv2.Canny(frontier_map, 100, 200)
#                     frontiers = np.copy(edges)
#                     frontier_goals = []
#                     frontier_goals_info_gains = []
#                     SEARCH_SIZE = 6  # 3
#                     SEARCH_STRIDE = 10  # 5
#                     for i in range(0, edges.shape[0] - SEARCH_STRIDE, SEARCH_STRIDE):
#                         for j in range(
#                             0, edges.shape[1] - SEARCH_STRIDE, SEARCH_STRIDE
#                         ):
#                             if edges[i, j] == 255:
#                                 obstacle_cells_nearby = 0
#                                 unknown_cells_nearby = 0
#                                 for k in range(-SEARCH_SIZE, SEARCH_SIZE + 1):
#                                     for l in range(-SEARCH_SIZE, SEARCH_SIZE + 1):
#                                         if k != 0 and l != 0:
#                                             if frontier_map[i + k, j + l] == 0:
#                                                 obstacle_cells_nearby += 1
#                                         if frontier_map[i + k, j + l] == 127:
#                                             unknown_cells_nearby += 1

#                                 if obstacle_cells_nearby < 2:
#                                     if self.visualize:
#                                         cv2.circle(frontier_map, (j, i), 8, 0, -1)
#                                         cv2.circle(frontier_map, (j, i), 8, 255, 1)
#                                     frontier_goals.append(
#                                         (j, i)
#                                     )  # goals in x,y cv2 coords
#                                     frontier_goals_info_gains.append(
#                                         unknown_cells_nearby
#                                     )

#                     # select next exploration goal (greedy)
#                     if len(frontier_goals) > 0:
#                         goal_dist = 0
#                         agent_pos = [
#                             obs[0]['position']['position'][2],
#                             obs[0]['position']['position'][0],
#                         ]
#                         new_goal = frontier_goals[0]

#                         # do not select a goal that is too close
#                         worst_info_gain = 0
#                         frontier_counter = 0
#                         # select goals farther than 1m and with higher info gain, if possible

#                         # Random selection of next goal
#                         n = np.random.randint(len(frontier_goals))
#                         new_goal = frontier_goals[n]

#                         # convert pixels to global map coords
#                         lower_bound, upper_bound = envs.call_at(idx, "get_map_bounds")
#                         grid_size = (
#                             abs(upper_bound[2] - lower_bound[2])
#                             / frontier_map.shape[0],
#                             abs(upper_bound[0] - lower_bound[0])
#                             / frontier_map.shape[1],
#                         )
#                         # lower_bounds are inverted, why?!
#                         realworld_x = lower_bound[0] + new_goal[0] * grid_size[0]
#                         realworld_y = lower_bound[2] + new_goal[1] * grid_size[1]
#                         goal_world = [
#                             realworld_x * map_scale,
#                             realworld_y * map_scale,
#                         ]  # goal in world cords (m)

#                         goal_dist = np.linalg.norm(
#                             np.array(agent_pos) - np.array(goal_world)
#                         )
#                         # print("FrontierBaseline - New exploration goal (m):", goal_world)

#                         if self.visualize:
#                             cv2.circle(
#                                 frontier_map,
#                                 (new_goal[0] + 2, new_goal[1] + 2),
#                                 5,
#                                 255,
#                                 -1,
#                             )

#                         ####### A* ######
#                         if self.replan_retries[idx] < 3:
#                             self.replan_retries[idx] += 1
#                             agent_pos = [
#                                 obs[0]['position']['position'][2],
#                                 obs[0]['position']['position'][0],
#                             ]
#                             grid_x = int((agent_pos[1] - lower_bound[0]) / grid_size[0])
#                             grid_y = int((agent_pos[0] - lower_bound[2]) / grid_size[1])
#                             start_pos = (grid_x, grid_y)
#                             goal_pos = (
#                                 new_goal[0],
#                                 new_goal[1],
#                             )  # in pixels, cv2 cords

#                             scale = 8
#                             draw_scale = 2
#                             img = allo_map * 255
#                             img = cv2.resize(
#                                 img,
#                                 (int(img.shape[1] / scale), int(img.shape[0] / scale)),
#                             )

#                             grid = Grid(img, draw_scale)
#                             # grid.clear_grid()
#                             grid.set_start_node(
#                                 (int(start_pos[1] / scale), int(start_pos[0] / scale))
#                             )
#                             grid.set_end_node(
#                                 (int(goal_pos[1] / scale), int(goal_pos[0] / scale))
#                             )  # in np cords
#                             grid.create_obstacles_from_img(img)
#                             # print("Start planning...")
#                             astar(grid)
#                             # print("Done planning.")
#                             astar_img = grid.draw()

#                             # extract sub-goals
#                             if grid.path is not None:
#                                 self.replan_retries[idx] = 0
#                                 self.sub_goals[idx] = []  # just to be sure...
#                                 self.sub_goals_to_draw[idx] = []
#                                 for i in range(0, len(grid.path), scale * 2):
#                                     if self.visualize:
#                                         sub_goal = (
#                                             grid.path[i][1] * draw_scale,
#                                             grid.path[i][0] * draw_scale,
#                                         )  # in cv2 coords again
#                                         self.sub_goals_to_draw[idx].append(
#                                             (
#                                                 grid.path[i][1] * scale,
#                                                 grid.path[i][0] * scale,
#                                             )
#                                         )
#                                         cv2.circle(
#                                             astar_img,
#                                             sub_goal,
#                                             4,
#                                             (255, i * 2, i * 2),
#                                             -1,
#                                         )

#                                     sub_goal = (
#                                         grid.path[i][1] * scale,
#                                         grid.path[i][0] * scale,
#                                     )  # in cv2 coords again
#                                     realworld_x = (
#                                         lower_bound[0] + sub_goal[0] * grid_size[0]
#                                     )
#                                     realworld_y = (
#                                         lower_bound[2] + sub_goal[1] * grid_size[1]
#                                     )
#                                     self.sub_goals[idx].append(
#                                         (realworld_x, realworld_y)
#                                     )
#                                 # remove first goal, it's too close to the agent
#                                 if self.sub_goals[idx]:
#                                     self.sub_goals[idx].pop(-1)
#                                     if self.visualize:
#                                         self.sub_goals_to_draw[idx].pop(-1)
#                                 # self.sub_goals = self.sub_goals[::-1]
#                                 # print("subgoals added:", self.sub_goals)
#                                 # print("agent is in",agent_pos)

#                                 self.next_goal = True

#                             if self.visualize:
#                                 cv2.imshow("astar_img", astar_img)

#                         else:  # if A* failed
#                             self.replan_retries[idx] = 0
#                             envs.call_at(
#                                 idx,
#                                 "set_goals",
#                                 {
#                                     "data": [
#                                         NavigationGoal(
#                                             position=[goal_world[0], 0, goal_world[1]]
#                                         )
#                                     ]
#                                 },
#                             )

#                         ####### end A* ######

#                     else:  # no new frontiers, reset episode
#                         # benvs.call_at(idx, "set_done", {"done": True}) # random

#                         envs.call_at(
#                             idx,
#                             "set_goals",
#                             {
#                                 "data": [
#                                     NavigationGoal(
#                                         position=[
#                                             self.current_observations[idx][0][
#                                                 'position'
#                                             ]['position'][0],
#                                             0,
#                                             self.current_observations[idx][0][
#                                                 'position'
#                                             ]['position'][1],
#                                         ]
#                                     )
#                                 ]
#                             },
#                         )

#                     if self.visualize:
#                         cv2.imshow("frontier_map", frontier_map)

#                 if self.visualize:
#                     rgb_map = maps.colorize_draw_agent_and_fit_to_height(
#                         info["top_down_map"],
#                         int(info['top_down_map']['map'].shape[0] / map_scale),
#                     )
#                     cv2.circle(
#                         rgb_map, (goal_to_draw[0], goal_to_draw[1]), 12, (255, 0, 0), 2
#                     )
#                     cv2.imshow(
#                         "rgb_map",
#                         cv2.resize(
#                             rgb_map,
#                             (int(rgb_map.shape[1] / 4), int(rgb_map.shape[0] / 4)),
#                         ),
#                     )
#                     cv2.waitKey(1)

#                 # save obs

#                 self.num_steps_done += 1
#                 if done:
#                     self.current_steps[idx] = 0

#             if self.num_steps_done % 10 == 0:
#                 print(f"Exploration at {int(self.percent_done() * 100)}%")

#         envs.close()
#         return sorted(generated_observations_paths)


@baseline_registry.register_trainer(name="frontierbaseline-v1")
class FrontierBaselinev1(Baseline):
    def __init__(self, config, exp_base_dir, **kwargs):
        super().__init__(config, exp_base_dir, GoalFollower, **kwargs)

    def compute_distance(self, a, b):
        return np.linalg.norm(b - a)

    def train(self):
        pass

    def _init_train(self):
        envs = super()._init_train()

        num_envs = envs.num_envs
        self.counter = 0
        self.visualize = False
        self.counter = [0] * num_envs
        self.sub_goals = [[] for _ in range(num_envs)]
        self.sub_goals_to_draw = [None] * num_envs
        self.sub_goals_counter = [0] * num_envs
        self.replan = [True] * num_envs
        self.replan_retries = [0] * num_envs
        self.got_new_plan = [False] * num_envs
        return envs

    def generate(self) -> None:
        envs = self._init_train()

        num_envs = envs.num_envs

        goal_to_draw = [0, 0]
        generated_observations_paths = []
        map_scale = 1.0

        while not self.is_done():
            self._step(envs)

            for idx in range(num_envs):
                obs = self.current_observations[idx]
                episode = envs.current_episodes()[idx]
                n_step = envs.call_at(idx, "get_step")
                self.counter[idx] += 1

                info = self.current_infos[idx]
                done = self.current_dones[idx]
                if done and n_step < 200:
                    print(f"Episode {episode} DONE before time?")
                episode = envs.current_episodes()[idx]

                if len(self.sub_goals[idx]) == 0:
                    self.replan[idx] = True

                # is it time to go to next subgoal (subgoals are already in m here)?

                if (
                    self.got_new_plan[idx] or self.current_steps[idx] % 20 == 0
                ) and len(self.sub_goals[idx]) > 0:
                    new_sub_goal = self.sub_goals[idx].pop(-1)
                    self.got_new_plan[idx] = False
                    print(
                        "picking next subgoal. Remaining subgoals=",
                        len(self.sub_goals[idx]),
                    )
                    envs.call_at(
                        idx,
                        "set_goals",
                        {"data": [NavigationGoal(position=new_sub_goal)]},
                    )

                if (
                    info is not None
                    and obs is not None
                    and self.replan[idx]
                    and self.counter[idx] > 20
                ):  # (info['distance_to_goal'] < 0.5 or self.counter[idx] > 30): # time to replan
                    # print("Replanning...")
                    # self.counter[idx] = 0
                    self.replan[idx] = False
                    self.replan_retries[idx] = 0

                    # self.counter[idx] = 0

                    print("Replanning....")
                    # allo_map = maps.get_top down_map_from_sim(self.sim, 1024, True, 0.05, 0)
                    allo_map = info["top_down_map"]["map"].copy()
                    allo_map[allo_map > 1] = 1
                    expl_map = info["top_down_map"]["fog_of_war_mask"].copy()

                    lower_bound, upper_bound = envs.call_at(idx, "get_map_bounds")

                    new_size = (
                        int(abs(upper_bound[0] - lower_bound[0]) / 0.025),
                        int(abs(upper_bound[2] - lower_bound[2]) / 0.025),
                    )

                    scaled_down_allo_map = cv2.resize(allo_map, new_size)
                    scaled_down_expl_map = cv2.resize(expl_map, new_size)

                    frontier_map = (
                        scaled_down_allo_map * 127 + scaled_down_expl_map * 127
                    )

                    # compute frontiers
                    edges = cv2.Canny(frontier_map, 100, 200)
                    frontiers = np.copy(edges)
                    frontier_goals = []
                    frontier_goals_info_gains = []
                    SEARCH_SIZE = 3  # 3
                    SEARCH_STRIDE = 5  # 5
                    for i in range(0, edges.shape[0] - SEARCH_STRIDE, SEARCH_STRIDE):
                        for j in range(
                            0, edges.shape[1] - SEARCH_STRIDE, SEARCH_STRIDE
                        ):
                            if edges[i, j] == 255:
                                obstacle_cells_nearby = 0
                                unknown_cells_nearby = 0
                                for k in range(-SEARCH_SIZE, SEARCH_SIZE + 1):
                                    for l in range(-SEARCH_SIZE, SEARCH_SIZE + 1):
                                        if k != 0 and l != 0:
                                            if frontier_map[i + k, j + l] == 0:
                                                obstacle_cells_nearby += 1
                                        if frontier_map[i + k, j + l] == 127:
                                            unknown_cells_nearby += 1

                                if obstacle_cells_nearby < 2:
                                    if self.visualize:
                                        cv2.circle(frontier_map, (j, i), 8, 0, -1)
                                        cv2.circle(frontier_map, (j, i), 8, 255, 1)
                                    frontier_goals.append(
                                        (j, i)
                                    )  # goals in x,y cv2 coords
                                    frontier_goals_info_gains.append(
                                        unknown_cells_nearby
                                    )

                    # select next exploration goal (greedy)
                    if len(frontier_goals) > 0:
                        print("frontiers found")

                        goal_dist = 0
                        agent_pos = [
                            obs[0]["position"]["position"][2],
                            obs[0]["position"]["position"][0],
                        ]
                        new_goal = frontier_goals[0]

                        # do not select a goal that is too close
                        worst_info_gain = 0
                        frontier_counter = 0
                        # select goals farther than 1m and with higher info gain, if possible
                        while (  # goal_dist < 1.0 and
                            frontier_counter < len(frontier_goals) - 1
                        ):
                            frontier_counter += 1
                            n = np.random.randint(len(frontier_goals))
                            new_goal = frontier_goals[n]
                            info_gain = frontier_goals_info_gains[n]
                            if info_gain < worst_info_gain:
                                continue
                            worst_info_gain = info_gain

                        # convert pixels to global map coords
                        grid_size = (
                            abs(upper_bound[2] - lower_bound[2])
                            / frontier_map.shape[0],
                            abs(upper_bound[0] - lower_bound[0])
                            / frontier_map.shape[1],
                        )
                        # lower_bounds are inverted, why?!
                        realworld_x = lower_bound[0] + new_goal[0] * grid_size[0]
                        realworld_y = lower_bound[2] + new_goal[1] * grid_size[1]
                        goal_world = [
                            realworld_x * map_scale,
                            realworld_y * map_scale,
                        ]  # goal in world cords (m)

                        goal_dist = np.linalg.norm(
                            np.array(agent_pos) - np.array(goal_world)
                        )

                        if self.visualize:
                            cv2.circle(
                                frontier_map,
                                (new_goal[0] + 2, new_goal[1] + 2),
                                5,
                                255,
                                -1,
                            )

                        if self.visualize:
                            cv2.imshow("frontier_map", frontier_map)

                        ####### Planning ######
                        if self.replan_retries[idx] < 3:
                            print("trying to plan #", self.replan_retries[idx])

                            self.replan_retries[idx] += 1
                            agent_pos = [
                                obs[0]["position"]["position"][2],
                                obs[0]["position"]["position"][0],
                            ]
                            grid_x = int((agent_pos[1] - lower_bound[0]) / grid_size[0])
                            grid_y = int((agent_pos[0] - lower_bound[2]) / grid_size[1])
                            start_pos = [grid_x, grid_y]
                            goal_pos = [
                                new_goal[0],
                                new_goal[1],
                            ]  # in pixels, cv2 cords

                            scale = 8
                            draw_scale = 2
                            scaled_down_allo_map = scaled_down_allo_map * 255

                            # actual A*

                            path, img = do_plan(
                                scaled_down_allo_map, start_pos, goal_pos, False, False
                            )

                            path.insert(0, goal_pos)
                            if self.visualize:
                                cv2.imshow("astar_img", img)

                            # extract sub-goals
                            self.sub_goals[idx] = []
                            for subgoal in path:
                                realworld_subgoal_x = lower_bound[0] + subgoal[0] * (
                                    abs(upper_bound[0] - lower_bound[0])
                                    / scaled_down_allo_map.shape[1]
                                )
                                realworld_subgoal_y = lower_bound[2] + subgoal[1] * (
                                    abs(upper_bound[2] - lower_bound[2])
                                    / scaled_down_allo_map.shape[0]
                                )
                                realworld_subgoal = [
                                    realworld_subgoal_x,
                                    obs[0]["position"]["position"][1],
                                    realworld_subgoal_y,
                                ]

                                self.sub_goals[idx].append(realworld_subgoal)

                            self.sub_goals[idx].pop(-1)

                            if len(self.sub_goals[idx]) > 0:
                                self.got_new_plan[idx] = True
                            else:
                                print("A* failed, no waypoints")

                        else:  # if A* failed
                            print("A* failed...")
                            self.replan_retries[idx] = 0
                            envs.call_at(
                                idx,
                                "set_goals",
                                {"data": [NavigationGoal(position=[0, 0, 0])]},
                            )

                        ####### end A* ######

                    else:  # no new frontiers, reset episode
                        print("No new frontiers, defaulting to goal in 0,0,0")
                        envs.call_at(
                            idx,
                            "set_goals",
                            {"data": [NavigationGoal(position=[0, 0, 0])]},
                        )
                        # envs.call_at(idx, "set_done", {"done": True})

                if self.visualize:
                    cv2.waitKey(5)

                # save obs
                for idx in range(num_envs):
                    obs = self.last_observations[idx]
                    obs[0]["goals"] = self.sub_goals[idx]
                    obs[0]["actions"] = self.last_actions[idx]

                    episode = envs.current_episodes()[idx]
                    n_step = envs.call_at(idx, "get_step")
                    generated_observations_paths.append(
                        save_obs(self.exp_path, episode.episode_id, obs, n_step)
                    )

                self.num_steps_done += 1
                if done:
                    self.current_steps[idx] = 0

        envs.close()
        return sorted(generated_observations_paths)


@deprecated("Use frontierbaseline-v1 instead")
@baseline_registry.register_trainer(name="frontierbaseline-v0")
class FrontierBaselinev0(Baseline):
    def __init__(self, config, exp_base_dir, **kwargs):
        super().__init__(config, exp_base_dir, GoalFollower, **kwargs)
        self.num_subgoals = 10

    def compute_distance(self, a, b):
        return np.linalg.norm(b - a)

    def train(self):
        pass

    def _init_train(self):
        envs = super()._init_train()

        num_envs = envs.num_envs
        self.counter = 0
        self.visualize = False
        self.counter = [0] * num_envs
        self.sub_goals = [None] * num_envs
        self.sub_goals_to_draw = [None] * num_envs
        self.sub_goals_counter = [0] * num_envs
        self.replan = [True] * num_envs
        self.replan_retries = [0] * num_envs
        return envs

    def generate(self) -> None:
        envs = self._init_train()

        num_envs = envs.num_envs

        goal_to_draw = [0, 0]
        generated_observations_paths = []
        # while not self.is_done():
        while not self.is_done():
            self._step(envs)

            for idx in range(num_envs):
                # save obs
                obs = self.current_observations[idx]
                episode = envs.current_episodes()[idx]
                step_ep = envs.call_at(idx, "get_step")
                generated_observations_paths.append(
                    save_obs(self.exp_path, episode.episode_id, obs, step_ep)
                )
            for idx in range(num_envs):
                self.counter[idx] += 1  # one counter per env?

                obs = self.current_observations[idx]
                info = self.current_infos[idx]
                done = self.current_dones[idx]
                episode = envs.current_episodes()[idx]

                map_scale = 1

                self.sub_goals_counter[idx] += 1

                if self.sub_goals_counter[idx] > self.num_subgoals:
                    self.sub_goals_counter[idx] = 0
                    if self.sub_goals[idx]:
                        new_sub_goal = self.sub_goals[idx].pop(-1)
                        if self.visualize:
                            goal_to_draw = self.sub_goals_to_draw[idx].pop(-1)
                        envs.call_at(
                            idx,
                            "set_goals",
                            {
                                "data": [
                                    NavigationGoal(
                                        position=[new_sub_goal[0], 0, new_sub_goal[1]]
                                    )
                                ]
                            },
                        )
                    else:
                        self.replan[idx] = True

                if (
                    info is not None
                    and obs is not None
                    and self.replan[idx]
                    and self.counter[idx] > 20
                ):  # (info['distance_to_goal'] < 0.5 or self.counter[idx] > 30): # time to replan
                    # print("Replanning...")
                    # self.counter[idx] = 0
                    self.replan[idx] = False
                    # allo_map = maps.get_top down_map_from_sim(self.sim, 1024, True, 0.05, 0)
                    allo_map = info["top_down_map"]["map"]
                    allo_map[allo_map > 1] = 1
                    expl_map = info["top_down_map"]["fog_of_war_mask"]
                    allo_map = cv2.resize(
                        allo_map,
                        (
                            int(allo_map.shape[1] / map_scale),
                            int(allo_map.shape[0] / map_scale),
                        ),
                    )
                    expl_map = cv2.resize(
                        expl_map,
                        (
                            int(expl_map.shape[1] / map_scale),
                            int(expl_map.shape[0] / map_scale),
                        ),
                    )
                    frontier_map = allo_map * 127 + expl_map * 127

                    # compute frontiers
                    edges = cv2.Canny(frontier_map, 100, 200)
                    frontiers = np.copy(edges)
                    frontier_goals = []
                    frontier_goals_info_gains = []
                    SEARCH_SIZE = 6  # 3
                    SEARCH_STRIDE = 10  # 5
                    for i in range(0, edges.shape[0] - SEARCH_STRIDE, SEARCH_STRIDE):
                        for j in range(
                            0, edges.shape[1] - SEARCH_STRIDE, SEARCH_STRIDE
                        ):
                            if edges[i, j] == 255:
                                obstacle_cells_nearby = 0
                                unknown_cells_nearby = 0
                                for k in range(-SEARCH_SIZE, SEARCH_SIZE + 1):
                                    for l in range(-SEARCH_SIZE, SEARCH_SIZE + 1):
                                        if k != 0 and l != 0:
                                            if frontier_map[i + k, j + l] == 0:
                                                obstacle_cells_nearby += 1
                                        if frontier_map[i + k, j + l] == 127:
                                            unknown_cells_nearby += 1

                                if obstacle_cells_nearby < 2:
                                    if self.visualize:
                                        cv2.circle(frontier_map, (j, i), 8, 0, -1)
                                        cv2.circle(frontier_map, (j, i), 8, 255, 1)
                                    frontier_goals.append(
                                        (j, i)
                                    )  # goals in x,y cv2 coords
                                    frontier_goals_info_gains.append(
                                        unknown_cells_nearby
                                    )

                    # select next exploration goal (greedy)
                    if len(frontier_goals) > 0:
                        goal_dist = 0
                        agent_pos = [
                            obs[0]["position"]["position"][2],
                            obs[0]["position"]["position"][0],
                        ]
                        new_goal = frontier_goals[0]

                        # do not select a goal that is too close
                        worst_info_gain = 0
                        frontier_counter = 0
                        # select goals farther than 1m and with higher info gain, if possible
                        while (
                            goal_dist < 1.0
                            and frontier_counter < len(frontier_goals) - 1
                        ):
                            frontier_counter += 1
                            n = np.random.randint(len(frontier_goals))
                            new_goal = frontier_goals[n]
                            info_gain = frontier_goals_info_gains[n]
                            if info_gain < worst_info_gain:
                                continue
                            worst_info_gain = info_gain

                        # convert pixels to global map coords
                        lower_bound, upper_bound = envs.call_at(idx, "get_map_bounds")
                        grid_size = (
                            abs(upper_bound[2] - lower_bound[2])
                            / frontier_map.shape[0],
                            abs(upper_bound[0] - lower_bound[0])
                            / frontier_map.shape[1],
                        )
                        # lower_bounds are inverted, why?!
                        realworld_x = lower_bound[0] + new_goal[0] * grid_size[0]
                        realworld_y = lower_bound[2] + new_goal[1] * grid_size[1]
                        goal_world = [
                            realworld_x * map_scale,
                            realworld_y * map_scale,
                        ]  # goal in world cords (m)

                        goal_dist = np.linalg.norm(
                            np.array(agent_pos) - np.array(goal_world)
                        )
                        # print("FrontierBaseline - New exploration goal (m):", goal_world)

                        if self.visualize:
                            cv2.circle(
                                frontier_map,
                                (new_goal[0] + 2, new_goal[1] + 2),
                                5,
                                255,
                                -1,
                            )

                        ####### A* ######
                        if self.replan_retries[idx] < 3:
                            self.replan_retries[idx] += 1
                            agent_pos = [
                                obs[0]["position"]["position"][2],
                                obs[0]["position"]["position"][0],
                            ]
                            grid_x = int((agent_pos[1] - lower_bound[0]) / grid_size[0])
                            grid_y = int((agent_pos[0] - lower_bound[2]) / grid_size[1])
                            start_pos = (grid_x, grid_y)
                            goal_pos = (
                                new_goal[0],
                                new_goal[1],
                            )  # in pixels, cv2 cords

                            scale = 8
                            draw_scale = 2
                            img = allo_map * 255
                            img = cv2.resize(
                                img,
                                (int(img.shape[1] / scale), int(img.shape[0] / scale)),
                            )

                            grid = Grid(img, draw_scale)
                            # grid.clear_grid()
                            grid.set_start_node(
                                (int(start_pos[1] / scale), int(start_pos[0] / scale))
                            )
                            grid.set_end_node(
                                (int(goal_pos[1] / scale), int(goal_pos[0] / scale))
                            )  # in np cords
                            grid.create_obstacles_from_img(img)
                            # print("Start planning...")
                            astar(grid)
                            # print("Done planning.")
                            astar_img = grid.draw()

                            # extract sub-goals
                            if grid.path is not None:
                                self.replan_retries[idx] = 0
                                self.sub_goals[idx] = []  # just to be sure...
                                self.sub_goals_to_draw[idx] = []
                                for i in range(0, len(grid.path), scale * 2):
                                    if self.visualize:
                                        sub_goal = (
                                            grid.path[i][1] * draw_scale,
                                            grid.path[i][0] * draw_scale,
                                        )  # in cv2 coords again
                                        self.sub_goals_to_draw[idx].append(
                                            (
                                                grid.path[i][1] * scale,
                                                grid.path[i][0] * scale,
                                            )
                                        )
                                        cv2.circle(
                                            astar_img,
                                            sub_goal,
                                            4,
                                            (255, i * 2, i * 2),
                                            -1,
                                        )

                                    sub_goal = (
                                        grid.path[i][1] * scale,
                                        grid.path[i][0] * scale,
                                    )  # in cv2 coords again
                                    realworld_x = (
                                        lower_bound[0] + sub_goal[0] * grid_size[0]
                                    )
                                    realworld_y = (
                                        lower_bound[2] + sub_goal[1] * grid_size[1]
                                    )
                                    self.sub_goals[idx].append(
                                        (realworld_x, realworld_y)
                                    )
                                # remove first goal, it's too close to the agent
                                if self.sub_goals[idx]:
                                    self.sub_goals[idx].pop(-1)
                                    if self.visualize:
                                        self.sub_goals_to_draw[idx].pop(-1)
                                # self.sub_goals = self.sub_goals[::-1]
                                # print("subgoals added:", self.sub_goals)
                                # print("agent is in",agent_pos)

                                self.next_goal = True

                            if self.visualize:
                                cv2.imshow("astar_img", astar_img)

                        else:  # if A* failed
                            self.replan_retries[idx] = 0
                            envs.call_at(
                                idx,
                                "set_goals",
                                {
                                    "data": [
                                        NavigationGoal(
                                            position=[goal_world[0], 0, goal_world[1]]
                                        )
                                    ]
                                },
                            )

                        ####### end A* ######

                    else:  # no new frontiers, reset episode
                        envs.call_at(idx, "set_done", {"done": True})  # v0

                    if self.visualize:
                        cv2.imshow("frontier_map", frontier_map)

                if self.visualize:
                    rgb_map = maps.colorize_draw_agent_and_fit_to_height(
                        info["top_down_map"],
                        int(info["top_down_map"]["map"].shape[0] / map_scale),
                    )
                    cv2.circle(
                        rgb_map, (goal_to_draw[0], goal_to_draw[1]), 12, (255, 0, 0), 2
                    )
                    cv2.imshow(
                        "rgb_map",
                        cv2.resize(
                            rgb_map,
                            (int(rgb_map.shape[1] / 4), int(rgb_map.shape[0] / 4)),
                        ),
                    )
                    cv2.waitKey(10)

                self.num_steps_done += 1
                if done:
                    self.current_steps[idx] = 0

            if self.num_steps_done % 10 == 0:
                print(f"Progress: {int(self.percent_done() * 100)}%")

        envs.close()
        return sorted(generated_observations_paths)


class ObserveObjectAgent(habitat.Agent):
    def __init__(self, success_distance: float, goal_sensor_uuid: str) -> None:
        # self.env = env
        # self.sim = self.env.habitat_env._sim
        self.tracking_object_center = None
        self.tracking_object_center_global = None

        self.tracking_action_direction = "RIGHT"
        self.tracking_action_index = 0
        self.tracking_object_class = None
        self.tracking_object_id = None
        self.already_observed = []

    def get_action(self, observations):
        if observations is not None:
            print(
                "Detected:",
                observations[0]["bbsgt"]["instances"].pred_classes.numpy(),
                "Tracking:",
                self.tracking_object_class,
            )

        self.sim.set_object_motion_type(habitat_sim.physics.MotionType.KINEMATIC, 0)

        # if not tracking anything
        if (
            self.tracking_object_center is None
        ):  # or self.tracking_object_class not in observations[0]['bbsgt']['instances'].pred_classes.numpy():
            if self.tracking_object_center is None:
                print("Not tracking. Looking for object to track.")
            else:
                print("Lost tracking of object. Looking for new object to track.")

            best_action = np.random.choice(
                [
                    HabitatSimActions.MOVE_FORWARD,
                    HabitatSimActions.TURN_LEFT,
                    HabitatSimActions.TURN_RIGHT,
                    HabitatSimActions.TURN_RIGHT,
                ]
            )

            # if at least one object is detected
            curr_id = 0
            if (
                observations is not None
                and len(observations[0]["bbsgt"]["instances"].infos) > 0
            ):
                for curr_id in range(len(observations[0]["bbsgt"]["instances"].infos)):
                    # if curr_id has not been observed before
                    if (
                        observations[0]["bbsgt"]["instances"].infos[curr_id][
                            "id_object"
                        ]
                        not in self.already_observed
                    ):
                        # get object center in global frame
                        object_center = observations[0]["bbsgt"]["instances"].infos[
                            curr_id
                        ]["center"]
                        # print('observing object at position', object_center)
                        # print(observations['bbsgt']['instances'].pred_classes[0].numpy())

                        # transform center to local frame
                        agent_pose = observations[0]["position"]
                        object_position = [
                            object_center[0],
                            0,
                            object_center[-1],
                        ]  # [object_center[0],object_center[2],object_center[-1]]
                        object_rotation = np.quaternion(0, 0, 0, 1)
                        agent_rotation = agent_pose["orientation"]
                        agent_position = agent_pose["position"]

                        # get object pose in agent reference frame
                        object_rot_in_agent_ref = (
                            agent_rotation.inverse() * object_rotation
                        )

                        vq = np.quaternion(0, 0, 0, 1)
                        vq.imag = object_position - agent_position
                        object_pos_in_agent_ref = (
                            agent_rotation.inverse() * vq * agent_rotation
                        ).imag
                        object_distance_from_agent = -object_pos_in_agent_ref[2]
                        print(
                            "observing object at position relative to agent",
                            object_pos_in_agent_ref,
                            "distance",
                            object_distance_from_agent,
                        )

                        # if the object is at a certain distance from the agent and centered in the camera frame, start tracking
                        if (
                            object_pos_in_agent_ref[0] > -1.0
                            and object_pos_in_agent_ref[0] < 1.0
                        ):
                            if (
                                object_distance_from_agent > 1.5
                                and object_distance_from_agent < 4.0
                            ):
                                self.tracking_object_center = object_pos_in_agent_ref
                                self.tracking_object_center_global = mn.Vector3(
                                    object_position
                                )
                                self.tracking_object_class = (
                                    observations[0]["bbsgt"]["instances"]
                                    .pred_classes[curr_id]
                                    .numpy()
                                )
                                self.tracking_object_id = observations[0]["bbsgt"][
                                    "instances"
                                ].infos[curr_id]["id_object"]
                                self.already_observed.append(self.tracking_object_id)
                                if object_pos_in_agent_ref[0] > 0.0:
                                    self.tracking_action_direction = "RIGHT"
                                else:
                                    self.tracking_action_direction = "LEFT"

                                self.tracking_action_index = 0

        else:  # is tracking object
            # execute next action in list
            # print("tracking action index", self.tracking_action_index)
            print(
                "Tracking object",
                self.tracking_object_id,
                "from the",
                self.tracking_action_direction,
                "- angle",
                self.tracking_action_index,
                "of 360",
            )
            # if self.tracking_action_direction == "RIGHT":
            #     tracking_action_list = self.tracking_right_action_list
            # else:
            #     tracking_action_list = self.tracking_left_action_list

            # action = tracking_action_list[self.tracking_action_index]
            # self.tracking_action_index += 1
            # if self.tracking_action_index == len(tracking_action_list):
            #     self.tracking_action_index = 0
            a = np.deg2rad(self.tracking_action_index)
            new_x = self.tracking_object_center_global[0] + 2.0 * np.cos(a)
            new_z = self.tracking_object_center_global[2] + 2.0 * np.sin(a)
            # new_a = np.arctan2(new_z,new_x)
            self.tracking_action_index += 1

            if self.tracking_action_index > 359:
                self.tracking_action_index = 0
                self.tracking_object_center = None
                self.tracking_object_class = None
                self.tracking_object_center_global = None
                return HabitatSimActions.MOVE_FORWARD

            agent = self.sim.get_agent(0)
            agent_state = habitat_sim.AgentState()
            agent_state.position = mn.Vector3(
                [new_x, agent_state.position[1], new_z]
            )  # world space
            # agent_state.rotation = mn.Quaternion.rotation(  mn.Rad(new_a) , mn.Vector3(0,1,0) )

            tangent_orientation_matrix = mn.Matrix4.look_at(
                agent_state.position,
                self.tracking_object_center_global,
                np.array([0, 1.0, 0]),
            )
            tangent_orientation_q = mn.Quaternion.from_matrix(
                tangent_orientation_matrix.rotation()
            )
            agent_state.rotation = utils.quat_from_magnum(tangent_orientation_q)

            agent.set_state(agent_state)

            # # Get agent state
            # agent_state = agent.get_state()

            best_action = HabitatSimActions.MOVE_FORWARD

        return best_action


class ObserveObjectDiscreteActionsAgent(RandomAgent):
    def __init__(self, success_distance: float, goal_sensor_uuid: str) -> None:
        super().__init__(success_distance, goal_sensor_uuid)
        self.turn_count = 0
        self.tracking_object_center = None  # (0,0,0)
        self.tracking_right_action_list = [
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.TURN_RIGHT,
        ]

        self.tracking_left_action_list = [
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.TURN_RIGHT,
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_LEFT,
        ]

        self.tracking_action_direction = "RIGHT"
        self.tracking_action_index = 0
        self.tracking_object_class = None

    def act(self, observations: Observations) -> Dict[str, Union[int, int]]:
        print(
            "Detected:",
            observations["bbsgt"]["instances"].pred_classes.numpy(),
            "Tracking:",
            self.tracking_object_class,
        )

        if (
            self.tracking_object_center is None
            or self.tracking_object_class
            not in observations["bbsgt"]["instances"].pred_classes.numpy()
        ):
            if self.tracking_object_center is None:
                print("Not tracking. Looking for object to track.")
            else:
                print("Lost tracking of object. Looking for new object to track.")

            action = np.random.choice(
                [
                    HabitatSimActions.MOVE_FORWARD,
                    HabitatSimActions.TURN_LEFT,
                    HabitatSimActions.TURN_RIGHT,
                    HabitatSimActions.TURN_RIGHT,
                    HabitatSimActions.TURN_RIGHT,
                ]
            )

            # if at least one object is detected
            if len(observations["bbsgt"]["instances"].infos) > 0:
                # get object center in global frame
                object_center = observations["bbsgt"]["instances"].infos[0]["center"]
                # print('observing object at position', object_center)
                # print(observations['bbsgt']['instances'].pred_classes[0].numpy())

                # transform center to local frame
                agent_pose = observations["position"]
                object_position = [
                    object_center[0],
                    object_center[2],
                    object_center[-1],
                ]
                object_rotation = np.quaternion(0, 0, 0, 1)
                agent_rotation = agent_pose["orientation"]
                agent_position = agent_pose["position"]

                # get object pose in agent reference frame
                object_rot_in_agent_ref = agent_rotation.inverse() * object_rotation

                vq = np.quaternion(0, 0, 0, 1)
                vq.imag = object_position - agent_position
                object_pos_in_agent_ref = (
                    agent_rotation.inverse() * vq * agent_rotation
                ).imag
                object_distance_from_agent = -object_pos_in_agent_ref[2]
                print(
                    "observing object at position relative to agent",
                    object_pos_in_agent_ref,
                    "distance",
                    object_distance_from_agent,
                )

                # if the object is at a certain distance from the agent and centered in the camera frame, start tracking
                if (
                    object_pos_in_agent_ref[0] > -1.0
                    and object_pos_in_agent_ref[0] < 1.0
                ):
                    if (
                        object_distance_from_agent > 1.5
                        and object_distance_from_agent < 4.0
                    ):
                        self.tracking_object_center = object_pos_in_agent_ref
                        self.tracking_object_class = (
                            observations["bbsgt"]["instances"].pred_classes[0].numpy()
                        )
                        if object_pos_in_agent_ref[0] > 0.0:
                            self.tracking_action_direction = "RIGHT"
                        else:
                            self.tracking_action_direction = "LEFT"

        else:  # is tracking object
            # execute next action in list
            print("Tracking object", self.tracking_action_direction)
            if self.tracking_action_direction == "RIGHT":
                tracking_action_list = self.tracking_right_action_list
            else:
                tracking_action_list = self.tracking_left_action_list

            action = tracking_action_list[self.tracking_action_index]
            self.tracking_action_index += 1
            if self.tracking_action_index == len(tracking_action_list):
                self.tracking_action_index = 0

        return {"action": action}


@baseline_registry.register_trainer(name="observeobjectdiscreteactionsbaseline")
class ObserveObjectDiscreteActionsBaseline(Baseline):
    def __init__(self, config, exp_base_dir, **kwargs):
        super().__init__(
            config, exp_base_dir, ObserveObjectDiscreteActionsAgent, **kwargs
        )


@baseline_registry.register_trainer(name="observeobjectbaseline")
class ObserveObjectContinuosActionsBaseline(Baseline):
    def __init__(self, config, exp_base_dir, **kwargs):
        super().__init__(config, exp_base_dir, ObserveObjectAgent, **kwargs)


@baseline_registry.register_trainer(name="randomgoalsbaseline")
class RandomGoalsBaseline(Baseline):
    def __init__(self, config, exp_base_dir, **kwargs):
        super().__init__(config, exp_base_dir, GoalFollower, **kwargs)
        self.config = config

        self.visualize = self.config.randomgoals.visualize  # visualize maps

    def goto_next_subgoal(self, idx):
        if (self.got_new_plan[idx] or self.current_steps[idx] % 10 == 0) and len(
            self.sub_goals[idx]
        ) > 0:
            # only select next sub-goal if the previous one was reached
            if self.prev_sub_goal[idx] is None or np.linalg.norm(self.prev_sub_goal[idx] - self.current_observations[idx][0]['position']['position']) <= 0.5:
                print("Last sub-goal was reached.")

                new_sub_goal = self.sub_goals[idx].pop(-1)
                self.prev_sub_goal[idx] = new_sub_goal
                self.got_new_plan[idx] = False
                print("Picking next subgoal. Remaining subgoals=", len(self.sub_goals[idx]))
                self.envs.call_at(
                    idx, "set_goals", {"data": [NavigationGoal(position=new_sub_goal)]}
                )
            else:
                print("Last sub-goal NOT reached, keeping last sub-goal.")

    def compute_new_goals(self):
        if (
            self.first_step
            or self.current_steps[0] % self.config.randomgoals.replanning_steps == 0
        ):
            print("Computing new goals...")
            self.first_step = False

            cpu_actions = np.random.uniform(size=(self.envs.num_envs, 2))

            # get the current map
            mymaps = [
                self.current_infos[idx]["top_down_map"]["map"].copy() * 255
                for idx in range(self.envs.num_envs)
            ]

            mymaps_sizes_pixels = [x.shape for x in mymaps]

            for action in cpu_actions:
                print("goal:", action)

            rescaled_pixel_goals = [
                [
                    int(
                        action[0] * mymaps_sizes_pixels[i][1]
                        # * self.current_observations[i][0]['disagreement_map'].shape[1]
                    ),
                    int(
                        action[1] * mymaps_sizes_pixels[i][0]
                        # * self.current_observations[i][0]['disagreement_map'].shape[0]
                    ),
                ]
                for i, action in enumerate(cpu_actions)
            ]
            # compute subgoals for each goal
            for idx, pixel_goal in enumerate(rescaled_pixel_goals):
                # convert pixels to global map coords
                lower_bound, upper_bound = self.envs.call_at(idx, "get_map_bounds")
                grid_resolution = (
                    abs(upper_bound[2] - lower_bound[2]) / mymaps_sizes_pixels[idx][0],
                    abs(upper_bound[0] - lower_bound[0]) / mymaps_sizes_pixels[idx][1],
                )

                # lower_bounds are inverted, why?!
                realworld_x = lower_bound[0] + pixel_goal[0] * grid_resolution[0]
                realworld_y = lower_bound[2] + pixel_goal[1] * grid_resolution[1]

                goal_world = [
                    realworld_x,
                    realworld_y,
                ]  # goal in world coords (m)

                agent_pos = [
                    self.current_observations[idx][0]["position"]["position"][2],
                    self.current_observations[idx][0]["position"]["position"][0],
                ]

                scaled_down_map = mymaps[idx]  # no rescalement required

                grid_size = (
                    abs(upper_bound[2] - lower_bound[2]) / scaled_down_map.shape[0],
                    abs(upper_bound[0] - lower_bound[0]) / scaled_down_map.shape[1],
                )

                grid_x = int((agent_pos[1] - lower_bound[0]) / grid_size[0])
                grid_y = int((agent_pos[0] - lower_bound[2]) / grid_size[1])

                # set start end end pose for A*
                start_pos = [grid_x, grid_y]
                end_pos = pixel_goal

                end_pos[0] = max(0, min(end_pos[0], scaled_down_map.shape[1] - 1))
                end_pos[1] = max(0, min(end_pos[1], scaled_down_map.shape[0] - 1))

                # extract a path to the goal (path is in opencv pixel coords)
                path, img = do_plan(
                    scaled_down_map, start_pos, end_pos, False, True, end_pos
                )

                if len(path) == 0:
                    print("No path!")
                    self.sub_goals[idx] = []
                    continue
                path.insert(0, end_pos)

                if self.visualize:
                    self.astar_img[idx] = img.copy()
                    cv2.circle(
                        self.astar_img[idx],
                        (end_pos[0], end_pos[1]),
                        20,
                        (255, 255, 0),
                        4,
                    )

                # extract sub-goals
                self.sub_goals[idx] = []
                for subgoal in path:
                    realworld_subgoal_x = lower_bound[0] + subgoal[0] * (
                        abs(upper_bound[0] - lower_bound[0]) / scaled_down_map.shape[1]
                    )
                    realworld_subgoal_y = lower_bound[2] + subgoal[1] * (
                        abs(upper_bound[2] - lower_bound[2]) / scaled_down_map.shape[0]
                    )
                    realworld_subgoal = [
                        realworld_subgoal_x,
                        self.current_observations[idx][0]["position"]["position"][1],
                        realworld_subgoal_y,
                    ]

                    self.sub_goals[idx].append(realworld_subgoal)

                self.sub_goals[idx].pop(-1)

                if len(self.sub_goals[idx]) > 0:
                    self.got_new_plan[idx] = True
                else:
                    print("A* failed, no waypoints")

    def generate(self) -> None:
        self.envs = self._init_train()
        generated_observations_paths = []
        self.exp_path = os.path.join(self.exp_path)

        self.sub_goals = [[] for _ in range(self.envs.num_envs)]
        self.sub_goals_counter = [0] * self.envs.num_envs
        self.got_new_plan = [False] * self.envs.num_envs
        self.replan_retries = [0] * self.envs.num_envs
        self.replan = [True] * self.envs.num_envs
        self.astar_img = [None] * self.envs.num_envs
        self.prev_sub_goal = [None] * self.envs.num_envs

        self.first_step = True

        # data generation loop
        while not self.is_done():
            # step all envs
            self._step(self.envs)

            not_ready = False
            # if no observations or no disagr map, exit
            if not self.current_infos or not self.current_observations:
                print("\n\n\nINFOS OR OBS NOT READY")
                not_ready = True

            # for each env, visualize and send next subgoal if needed
            for idx in range(self.envs.num_envs):
                obs = self.current_observations[idx]
                info = self.current_infos[idx]
                episode = self.envs.current_episodes()[idx]

                # visualize maps for debugging purposes
                if self.visualize and self.astar_img[idx] is not None:
                    cv2.imshow("Env " + str(idx) + " A*", self.astar_img[idx])

                if self.visualize and ("disagreement_map" in obs[0]):
                    rgb_map = maps.colorize_draw_agent_and_fit_to_height(
                        info["top_down_map"], obs[0]["disagreement_map"].shape[0]
                    )
                    cv2.imshow("Env " + str(idx) + " map", rgb_map)
                    cv2.imshow(
                        "Env " + str(idx) + " disagreement", obs[0]["disagreement_map"]
                    )

                # if time to go to next subgoal, do it
                self.goto_next_subgoal(idx)

            if self.visualize:
                cv2.waitKey(10)

            # if time to predict new goal
            self.compute_new_goals()

            for idx in range(self.envs.num_envs):
                obs = self.current_observations[idx]
                obs[0]["goals"] = self.sub_goals[idx]
                episode = self.envs.current_episodes()[idx]
                n_step = self.envs.call_at(idx, "get_step")
                paths = save_obs(self.exp_path, episode.episode_id, obs, n_step)
                generated_observations_paths.append(paths)

            for idx in range(self.envs.num_envs):
                done = self.current_dones[idx]
                self.num_steps_done += 1
                if done:
                    self.current_steps[idx] = 0

            if self.num_steps_done % 10 == 0:
                print(f"Progress: {int(self.percent_done() * 100)}%")
        del self.current_dones
        del self.current_observations

        gc.collect()
        self.envs.close()
        return sorted(generated_observations_paths)


class SubGoalFollower(GoalFollower):
    def __init__(self, success_distance: float, goal_sensor_uuid: str) -> None:
        super().__init__(success_distance, goal_sensor_uuid)
        self.pos_th = self.dist_threshold_to_stop
        self.angle_th = float(np.deg2rad(15))
        self.random_prob = 0
        self.sub_goals = {} # dict where keys are the envs index and values are sub-goal position for each envs 
        self._goal_format = "POLAR"
        self._dimensionality = 4


    def _compute_pointgoal(
        self, source_position, source_rotation, goal_position
    ):
        direction_vector = goal_position - source_position
        direction_vector_agent = quaternion_rotate_vector(
            source_rotation.inverse(), direction_vector
        )

        if self._goal_format == "POLAR":
            if self._dimensionality == 2:
                rho, phi = cartesian_to_polar(
                    -direction_vector_agent[2], direction_vector_agent[0]
                )
                return np.array([rho, -phi], dtype=np.float32)
            else:
                _, phi = cartesian_to_polar(
                    -direction_vector_agent[2], direction_vector_agent[0]
                )
                theta = np.arccos(
                    direction_vector_agent[1]
                    / np.linalg.norm(direction_vector_agent)
                )
                rho = np.linalg.norm(direction_vector_agent)

                return np.array([rho, -phi, theta], dtype=np.float32)
        else:
            if self._dimensionality == 2:
                return np.array(
                    [-direction_vector_agent[2], direction_vector_agent[0]],
                    dtype=np.float32,
                )
            else:
                return direction_vector_agent

    def compute_goal_distance(self, agent_position):
        return np.linalg.norm(agent_position['position'] - self.sub_goal.position)

    def compute_goal_angle(self, agent_position):
        direction_vector = agent_position['position'] - self.sub_goal.position
        direction_vector_agent = quaternion_rotate_vector(agent_position['orientation'], direction_vector)
        phi = np.arctan2(-direction_vector_agent[2], direction_vector_agent[0])
        return -phi


    def act(self, observations: Observations, agent_position, env_idx) -> Dict[str, int]:
        sub_goal = self.sub_goals.get(env_idx, None)
        
        if sub_goal is not None:
            goal = self._compute_pointgoal(agent_position['position'], agent_position['orientation'], sub_goal.position)
            if goal[0] < 0.2:
                action = HabitatSimActions.stop
            else:
                angle_to_goal = self.normalize_angle(
                    np.array(goal[1])
                )
                if abs(angle_to_goal) < self.angle_th:
                    action = HabitatSimActions.move_forward
                else:
                    action = self.turn_towards_goal(angle_to_goal)
        else:
            action = HabitatSimActions.stop
        return {"action": action}

