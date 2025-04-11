# type: ignore
import copy
import cv2
import math
import numpy as np
import os
import time
from typing import *

from habitat.config.default import get_config
from habitat.tasks.nav.nav import NavigationGoal, NavigationTask
from habitat_baselines.agents.simple_agents import GoalFollower
from habitat_baselines.common.baseline_registry import baseline_registry
import open_clip
from PIL import Image
from sentence_transformers import SentenceTransformer
import torch
from experimenting_env.utils.sensors_utils import save_obs
from experimenting_env.agents.baselines import SubGoalFollower
from ..baselines import Baseline
from experimenting_env.agents.goal_exploration.goal_exploration import ObjectDetectorEnv
from experimenting_env.captioner.utils.utils import Configuration
from experimenting_env.captioner.models.coca.coca import CoCa
from experimenting_env.captioner.models.blip2.blip2 import BLIP2


@baseline_registry.register_trainer(name="frontierbaseline-v2")
class FrontierBaselinev2(Baseline):
    def __init__(self, config, exp_base_dir, detectron_args, **kwargs):
        super().__init__(config, exp_base_dir, SubGoalFollower, **kwargs)
        self.device = torch.device("cuda:0" if self.config.ppo.cuda else "cpu")
        self.device_config = self.config.device_config
        self.object_detector = ObjectDetectorEnv({"cfg": detectron_args}, device=self.device)
        self.captioner = self.get_captioner(self.config.captioner)

        self.captioner.to(self.device_config.captioner.device)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2").to(self.device)

    def _step(self, envs):
        super()._step(envs)
        self.compute_embeddings()
        
    def get_captioner(self, cfg):
        """ Get the captioner model based on the configuration settings"""
        if cfg.arch_name == "coca": 
            captioner_cfg = Configuration(arch_name=cfg.arch_name, model_name=cfg.model_name,
                                            checkpoint_name=cfg.checkpoint_name, height=cfg.height, width=cfg.width) 
            captioner = CoCa(captioner_cfg.captioner)

        elif cfg.arch_name == "blip2":
            captioner_cfg = Configuration(arch_name=cfg.arch_name, model_name=cfg.model_name,
                                            height=cfg.height, width=cfg.width) 
            captioner = BLIP2(captioner_cfg.captioner)

        return captioner.eval()

    def compute_distance(self, a, b):
        return np.linalg.norm(b - a)

    def train(self):
        pass

    def _init_train(self):
        envs = super()._init_train()
        # init parameters for generate
        self.visualize = self.config.ppo.visualize
        self.replanning_time = [True] * envs.num_envs
        self.sub_goals = [[] for _ in range(envs.num_envs)]
        self.goal = [None] * envs.num_envs
        self.got_new_plan = [False] * envs.num_envs
        self.frontier_maps = [None] * envs.num_envs
        return envs

    # return frontier points and associated information gain
    def compute_frontiers(self, obs, info, idx):
        allocentric_map = info["top_down_map"]["map"].copy()
        allocentric_map[allocentric_map > 1] = 1
        exploration_map = info["top_down_map"]["fog_of_war_mask"].copy()

        lower_bound, upper_bound = self.envs.call_at(idx, "get_upper_and_lower_map_bounds")

        frontier_map = allocentric_map * 127 + exploration_map * 127

        # compute frontiers
        frontier_goals = []
        frontier_goals_info_gains = []

        edges = cv2.Canny(frontier_map, 100, 200)
        search_size, search_stride = 6, 5

        for i in range(0, edges.shape[0] - search_stride, search_stride):
            for j in range(0, edges.shape[1] - search_stride, search_stride):
                if edges[i, j] == 255:
                    obstacle_cells_nearby = 0
                    unknown_cells_nearby = 0
                    for k in range(-search_size, search_size + 1):
                        for l in range(-search_size, search_size + 1):
                            if k != 0 and l != 0:
                                if frontier_map[i + k, j + l] == 0:
                                    obstacle_cells_nearby += 1
                            if frontier_map[i + k, j + l] == 127:
                                unknown_cells_nearby += 1

                    if obstacle_cells_nearby < 2:
                        frontier_goals.append(
                            (j, i)
                        )  # frontier goals are in (x,y) cv2 coords
                        frontier_goals_info_gains.append(
                            unknown_cells_nearby
                        )  # trivial information gain = unknown neighbour cells

        # save frontier image for debugging
        self.frontier_maps[idx] = cv2.cvtColor(frontier_map, cv2.COLOR_GRAY2BGR)
        for frontier in frontier_goals:
            cv2.circle(self.frontier_maps[idx], frontier, 20, (0, 0, 255), 4)

        return frontier_goals, frontier_goals_info_gains

    def compute_embeddings(self, batch_size=8):
        for idx in range(0, self.envs.num_envs, batch_size):
            max_idx = min(self.envs.num_envs, idx + batch_size)
            images = [self.current_observations[i]["rgb"] for i in range(idx, max_idx)]
            preds = self.object_detector.predict_batch(images)

            for i in range(max_idx - idx):
                current_env = idx + i
                embeddings = [] 
                captions = []
                gt_logits_list = []

                env_scene = self.envs.call_at(0, "get_scene")
                self.current_observations[current_env]['bbs'] = preds[i]
                for j in range(len(self.current_observations[current_env]['bbs']['instances'].pred_boxes)): 
                    x, y, w, h = self.current_observations[current_env]['bbs']['instances'].pred_boxes[:].tensor[j]
                    x, y, w, h = int(x.item()), int(y.item()), int(w.item()), int(h.item())
                    # TODO: check if the coordinate for the cropping are correct
                    im = Image.fromarray(self.current_observations[current_env]['rgb'][y:h, x:w], 'RGB')
                    out = self.captioner(im)
                    caption = out['text']
                    generated_embeddings = self.encoder.encode(caption)
                    embeddings.append(torch.tensor(generated_embeddings))
                    captions.append(caption)
                    print(caption)
                    # Create logits for the gt detections (all 0 except for the gt class)
                    #gt_logits = torch.zeros(7)
                    #gt_class = self.current_observations[current_env]['bbs']['instances'].pred_classes[j]
                    # mp3d classes goes from 56 to 62 
                    #gt_logits[gt_class - 56] = 1 
                    #gt_logits_list.append(gt_logits)
                if len(embeddings) == 0:
                    self.current_observations[current_env]['bbs']['instances'].embeddings = torch.tensor(())
                    self.current_observations[current_env]['bbs']['instances'].captions = []
                else:
                    self.current_observations[current_env]['bbs']['instances'].embeddings = torch.stack(embeddings[:])
                    self.current_observations[current_env]['bbs']['instances'].captions = captions

                # if len(gt_logits_list) != 0:
                #     self.current_observations[current_env]['bbs']['instances'].gt_logits = torch.stack(gt_logits_list, dim=0)
                # else:
                #     self.current_observations[current_env]['bbs']['instances'].gt_logits = torch.tensor(())

                # self.current_observations[current_env]['position'] = self.envs.call_at(current_env, "get_agent_position")

                #self.current_observations[current_env]['bbs'] = {'instances': copy.deepcopy(self.current_observations[current_env]['bbsgt']['instances'])}
                
                self.current_observations[current_env]['position'] = self.envs.call_at(current_env, "get_agent_position") 
                episode_id = self.envs.call_at(current_env, "get_episode_id")
                map_bounds = self.envs.call_at(current_env, "get_upper_and_lower_map_bounds")
                self.envs.call_at(
                    current_env,
                    "update_pointcloud",
                    {"observations": self.current_observations[current_env], "episode_id":episode_id, "map_bounds":map_bounds},
                )
                disagreement_map = self.envs.call_at(
                    current_env, 
                    "get_and_update_disagreement_map",
                    {"map_bounds": map_bounds}
                )

                self.current_observations[current_env][
                    'disagreement_map'
                ] = disagreement_map
                
     # visualize frontier map, goal and subgoals for debugging
    def visualize_map(self, obs, info, idx, downsize=0.5):
        frontier_map_rgb = self.frontier_maps[idx]
        if frontier_map_rgb is None:
            return

        agent_position = (
            obs["position"]["position"][2],
            obs["position"]["position"][0],
        )
        agent_position_pixels = self.coords2pixels(agent_position, info, idx)
        cv2.circle(frontier_map_rgb, tuple(agent_position_pixels), 14, (0, 255, 0), -1)

        if self.goal[idx] is not None:
            goal = [self.goal[idx][2], self.goal[idx][0]]
            goal_pixels = self.coords2pixels(goal, info, idx)
            cv2.circle(frontier_map_rgb, tuple(goal_pixels), 20, (0, 0, 255), -1)

            counter = 0
            for sub_goal, next_sub_goal in zip(
                self.sub_goals[idx][:-1], self.sub_goals[idx][1:]
            ):
                sub_goal = [sub_goal[2], sub_goal[0]]
                next_sub_goal = [next_sub_goal[2], next_sub_goal[0]]
                sub_goal_pixels = self.coords2pixels(sub_goal, info, idx)
                next_sub_goal_pixels = self.coords2pixels(next_sub_goal, info, idx)

                cv2.circle(
                    frontier_map_rgb,
                    tuple(next_sub_goal_pixels),
                    14,
                    (255, counter, counter),
                    -1,
                )
                cv2.line(
                    frontier_map_rgb,
                    tuple(sub_goal_pixels),
                    tuple(next_sub_goal_pixels),
                    (255, counter, counter),
                    4,
                )
                counter += 80

        frontier_map_rgb = cv2.resize(
            frontier_map_rgb,
            (
                int(frontier_map_rgb.shape[1] * downsize),
                int(frontier_map_rgb.shape[0] * downsize),
            ),
        )
        if self.visualize:
            cv2.imshow("frontier_map", frontier_map_rgb)
            im = obs['rgb'].copy()
            if len(self.current_observations[0]['bbs']['instances'].pred_boxes)!= 0: 
                for j in range(len(self.current_observations[0]['bbs']['instances'].pred_boxes)): 
                    x, y, w, h = self.current_observations[0]['bbs']['instances'].pred_boxes[:].tensor[j]
                    x, y, w, h = int(x.item()), int(y.item()), int(w.item()), int(h.item())
                    cv2.rectangle(im, (x, y), (w, h), (0, 255, 0))
                    cv2.putText(im, self.current_observations[0]['bbs']['instances'].captions[j], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 2)
                    
            cv2.imshow("rgb", im)
            cv2.waitKey(5)

            cv2.imshow("Env " + str(idx) + " disagreement", self.current_observations[0]["disagreement_map"])

    # select next frontier among frontier_goals based on some heuristic (greedy for now)
    def select_next_frontier(self, obs, frontier_goals, frontier_goals_info_gains):
        new_goal = frontier_goals[0]
        worst_info_gain = 0
        frontier_counter = 0

        # what is this?!
        while frontier_counter < len(frontier_goals) - 1:
            frontier_counter += 1
            n = np.random.randint(len(frontier_goals))
            new_goal = frontier_goals[n]
            new_info_gain = frontier_goals_info_gains[n]
            if new_info_gain < worst_info_gain:
                continue
            worst_info_gain = new_info_gain

        return new_goal

    # convert point in cv2 pixel coords to habitat 3D coords in (m)
    def pixels2coords(self, pixels, info, idx):
        lower_bound, upper_bound = self.envs.call_at(idx, "get_upper_and_lower_map_bounds")
        grid_size = (
            abs(upper_bound[2] - lower_bound[2]) / info["top_down_map"]["map"].shape[0],
            abs(upper_bound[0] - lower_bound[0]) / info["top_down_map"]["map"].shape[1],
        )
        # lower_bounds are inverted, why?!
        realworld_x = (
            lower_bound[0] + pixels[0] * grid_size[0]
        )  # goal in world cords (m)
        realworld_y = lower_bound[2] + pixels[1] * grid_size[1]

        return np.array([realworld_x, 0, realworld_y])

    # convert habitat 3D coords in (m) to cv2 pixel coords
    def coords2pixels(self, coords, info, idx):
        lower_bound, upper_bound = self.envs.call_at(idx, "get_upper_and_lower_map_bounds")
        grid_size = (
            abs(upper_bound[2] - lower_bound[2]) / info["top_down_map"]["map"].shape[0],
            abs(upper_bound[0] - lower_bound[0]) / info["top_down_map"]["map"].shape[1],
        )
        grid_x = int((coords[1] - lower_bound[0]) / grid_size[0])
        grid_y = int((coords[0] - lower_bound[2]) / grid_size[1])

        return [grid_x, grid_y]

    def generate(self) -> None:
        self.envs = self._init_train()
        generated_observations_paths = []

        while not self.is_done():
            self._step(self.envs)

            for idx in range(self.envs.num_envs):
                obs = self.current_observations[idx]
                info = self.current_infos[idx]
                done = self.current_dones[idx]
                episode = self.envs.current_episodes()[idx]
                if done:
                    self.replanning_time = [True] * self.envs.num_envs
                    self.sub_goals = [[] for _ in range(self.envs.num_envs)]
                    self.goal = [None] * self.envs.num_envs
                    self.previous_sub_goal = [None] * self.envs.num_envs
                    self.elapsed_since_last_sub_goal = [0] * self.envs.num_envs
                    self.got_new_plan = [False] * self.envs.num_envs
                    self.frontier_maps = [None] * self.envs.num_envs

                if self.visualize:
                    self.visualize_map(obs, info, idx)

                # save obs
                #n_step = self.envs.call_at(idx, "get_step")
                n_step = self.current_steps[idx]
                generated_observations_paths.append(
                    save_obs(self.exp_path, episode.episode_id, obs, n_step)
                )

                # TODO: check if last frontier has been reached, then replan
                # if self.num_steps_done % 10 == 0:
                #     self.replanning_time[idx] = True
                if self.num_steps_done % 20 == 0 and len(self.sub_goals[idx]) == 0:
                    self.replanning_time[idx] = True

                # check if time to go to next subgoal (either a new goal is available, or 20 steps elapsed)
                if (self.got_new_plan[idx] or self.num_steps_done % 20 == 0) and len(
                    self.sub_goals[idx]
                ) > 0:
                    new_sub_goal = self.sub_goals[idx].pop(0)
                    self.got_new_plan[idx] = False
                    print(
                        "picking next subgoal. Remaining subgoals=",
                        len(self.sub_goals[idx]),
                    )
                    self.envs.call_at(
                        idx,
                        "set_goals",
                        {"data": [NavigationGoal(position=new_sub_goal)]},
                    )
                    self.agent.sub_goals[idx] = NavigationGoal(position=new_sub_goal)

                # if it's time to replan
                if self.replanning_time[idx]:
                    self.replanning_time[idx] = False

                    frontier_goals, frontier_goals_info_gains = self.compute_frontiers(
                        obs, info, idx
                    )

                    # greedily select next exploration goal among frontier goals and compute subgoals using pathfinder
                    if len(frontier_goals) > 0:
                        agent_position = obs["position"]["position"]

                        new_goal_pixels = self.select_next_frontier(
                            obs, frontier_goals, frontier_goals_info_gains
                        )
                        new_goal_coords = self.pixels2coords(new_goal_pixels, info, idx)
                        new_goal_coords[1] = agent_position[1]

                        self.goal[idx] = new_goal_coords

                        print(
                            "New frontier goal at:", new_goal_coords, "- sending agent"
                        )

                        path_points = self.envs.call_at(
                            idx,
                            "get_path",
                            {"agent_position": agent_position, "goal": new_goal_coords},
                        )
                        # print("Path:", path_points)
                        self.sub_goals[idx] = path_points.tolist()
                        self.got_new_plan[idx] = (
                            True  # signal that a new plan is available
                        )

                    else:  # if no new frontiers found, map is fully explored -> reset episode
                        print("No new frontiers, sending agent back to origin (0, 0)")
                        self.envs.call_at(
                            idx,
                            "set_goals",
                            {
                                "data": [
                                    NavigationGoal(
                                        position=[
                                            0,
                                            obs["position"]["position"][1],
                                            0,
                                        ]
                                    )
                                ]
                            },
                        )
                        self.got_new_plan[idx] = False
                self.num_steps_done += 1

        self.envs.close()

        return sorted(generated_observations_paths)


# similar to frontierbaseline-v2, but new subgoals are sent upon reaching the previous one
@baseline_registry.register_trainer(name="frontierbaseline-v3")
class FrontierBaselinev3(Baseline):
    def __init__(self, config, exp_base_dir, detectron_args, **kwargs):
        super().__init__(config, exp_base_dir, SubGoalFollower, **kwargs)
        self.device = torch.device("cuda:0" if self.config.ppo.cuda else "cpu")
        self.device_config = self.config.device_config
        self.object_detector = ObjectDetectorEnv({"cfg": detectron_args}, device=self.device)
        self.captioner = self.get_captioner(self.config.captioner)

        self.captioner.to(self.device_config.captioner.device)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2").to(self.device)
        
    def _step(self, envs):
        super()._step(envs)
        self.compute_embeddings()
        
    def get_captioner(self, cfg):
        """ Get the captioner model based on the configuration settings"""
        if cfg.arch_name == "coca": 
            captioner_cfg = Configuration(arch_name=cfg.arch_name, model_name=cfg.model_name,
                                            checkpoint_name=cfg.checkpoint_name, height=cfg.height, width=cfg.width) 
            captioner = CoCa(captioner_cfg.captioner)

        elif cfg.arch_name == "blip2":
            captioner_cfg = Configuration(arch_name=cfg.arch_name, model_name=cfg.model_name,
                                            height=cfg.height, width=cfg.width) 
            captioner = BLIP2(captioner_cfg.captioner)

        return captioner.eval()
    
    def compute_embeddings(self, batch_size=8):
        for idx in range(0, self.envs.num_envs, batch_size):
            max_idx = min(self.envs.num_envs, idx + batch_size)
            images = [self.current_observations[i]["rgb"] for i in range(idx, max_idx)]
            preds = self.object_detector.predict_batch(images)

            for i in range(max_idx - idx):
                current_env = idx + i
                embeddings = [] 
                captions = []
                gt_logits_list = []

                env_scene = self.envs.call_at(0, "get_scene")
                self.current_observations[current_env]['bbs'] = preds[0][i]
                for j in range(len(self.current_observations[current_env]['bbs']['instances'].pred_boxes)): 
                    x, y, w, h = self.current_observations[current_env]['bbs']['instances'].pred_boxes[:].tensor[j]
                    x, y, w, h = int(x.item()), int(y.item()), int(w.item()), int(h.item())
                    # TODO: check if the coordinate for the cropping are correct
                    im = Image.fromarray(self.current_observations[current_env]['rgb'][y:h, x:w], 'RGB')
                    out = self.captioner(im)
                    caption = out['text']
                    generated_embeddings = self.encoder.encode(caption)
                    embeddings.append(torch.tensor(generated_embeddings))
                    captions.append(caption)
                    print(caption)
                    # Create logits for the gt detections (all 0 except for the gt class)
                    #gt_logits = torch.zeros(7)
                    #gt_class = self.current_observations[current_env]['bbs']['instances'].pred_classes[j]
                    # mp3d classes goes from 56 to 62 
                    #gt_logits[gt_class - 56] = 1 
                    #gt_logits_list.append(gt_logits)
                if len(embeddings) == 0:
                    self.current_observations[current_env]['bbs']['instances'].embeddings = torch.tensor(())
                    self.current_observations[current_env]['bbs']['instances'].captions = []
                else:
                    self.current_observations[current_env]['bbs']['instances'].embeddings = torch.stack(embeddings[:])
                    self.current_observations[current_env]['bbs']['instances'].captions = captions

                # if len(gt_logits_list) != 0:
                #     self.current_observations[current_env]['bbs']['instances'].gt_logits = torch.stack(gt_logits_list, dim=0)
                # else:
                #     self.current_observations[current_env]['bbs']['instances'].gt_logits = torch.tensor(())

                # self.current_observations[current_env]['position'] = self.envs.call_at(current_env, "get_agent_position")

                #self.current_observations[current_env]['bbs'] = {'instances': copy.deepcopy(self.current_observations[current_env]['bbsgt']['instances'])}
                
                self.current_observations[current_env]['position'] = self.envs.call_at(current_env, "get_agent_position") 
                episode_id = self.envs.call_at(current_env, "get_episode_id")
                map_bounds = self.envs.call_at(current_env, "get_upper_and_lower_map_bounds")
                self.envs.call_at(
                    current_env,
                    "update_pointcloud",
                    {"observations": self.current_observations[current_env], "episode_id":episode_id, "map_bounds":map_bounds},
                )
                disagreement_map = self.envs.call_at(
                    current_env, 
                    "get_and_update_disagreement_map",
                    {"map_bounds": map_bounds}
                )

                self.current_observations[current_env][
                    'disagreement_map'
                ] = disagreement_map

    def compute_distance(self, a, b):
        return np.linalg.norm(b - a)

    def train(self):
        pass

    def _init_train(self):
        envs = super()._init_train()
        # init parameters for generate
        self.visualize = self.config.ppo.visualize
        self.replanning_time = [True] * envs.num_envs
        self.sub_goals = [[] for _ in range(envs.num_envs)]
        self.goal = [None] * envs.num_envs
        self.previous_sub_goal = [None] * envs.num_envs
        self.elapsed_since_last_sub_goal = [0] * envs.num_envs
        self.got_new_plan = [False] * envs.num_envs
        self.frontier_maps = [None] * envs.num_envs
        self.edge_maps = [None] * envs.num_envs
        # self.episode_elapsed_steps = [0] * envs.num_envs

        return envs
    
    

    # return frontier points and associated information gain
    def compute_frontiers(self, obs, info, idx):
        allocentric_map = info["top_down_map"]["map"].copy()
        allocentric_map[allocentric_map > 1] = 1
        exploration_map = info["top_down_map"]["fog_of_war_mask"].copy()

        lower_bound, upper_bound = self.envs.call_at(idx, "get_upper_and_lower_map_bounds")

        frontier_map = allocentric_map * 127 + exploration_map * 127

        # save frontier image for debugging
        self.frontier_maps[idx] = cv2.cvtColor(frontier_map, cv2.COLOR_GRAY2BGR)

        # compute frontiers
        frontier_goals = []
        frontier_goals_info_gains = []

        edges = cv2.Canny(frontier_map, 100, 200)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel)
        self.edge_maps[idx] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        search_size, search_stride = 20, 5

        for i in range(search_size, edges.shape[0] - search_size, search_stride):
            for j in range(search_size, edges.shape[1] - search_size, search_stride):
                # cv2.circle(self.frontier_maps[idx], (j,i), 1, (0,255,0), -1)

                if edges[i, j] == 255:
                    obstacle_cells_nearby = 0
                    unknown_cells_nearby = 0
                    free_cells_nearby = 0
                    for k in range(-search_size, search_size + 1):
                        for l in range(-search_size, search_size + 1):
                            # if k != 0 and l != 0:
                            if frontier_map[i + k, j + l] == 0:
                                obstacle_cells_nearby += 1
                            if frontier_map[i + k, j + l] == 127:
                                unknown_cells_nearby += 1
                            if frontier_map[i + k, j + l] == 255:
                                free_cells_nearby += 1

                    if obstacle_cells_nearby < 3:  # > 2 and unknown_cells_nearby > 2:
                        frontier_goals.append(
                            (j, i)
                        )  # frontier goals are in (x,y) cv2 coords
                        frontier_goals_info_gains.append(
                            unknown_cells_nearby
                        )  # trivial information gain = unknown neighbour cells

        for frontier in frontier_goals:
            cv2.circle(self.frontier_maps[idx], frontier, 20, (0, 0, 255), 4)

        return frontier_goals, frontier_goals_info_gains

    # visualize frontier map, goal and subgoals for debugging
    def visualize_map(self, obs, info, idx, downsize=0.5):
        frontier_map_rgb = self.frontier_maps[idx]
        if frontier_map_rgb is None:
            return

        agent_position = (
            obs["position"]["position"][2],
            obs["position"]["position"][0],
        )
        agent_position_pixels = self.coords2pixels(agent_position, info, idx)
        cv2.circle(frontier_map_rgb, tuple(agent_position_pixels), 14, (0, 255, 0), -1)

        if self.goal[idx] is not None:
            goal = [self.goal[idx][2], self.goal[idx][0]]
            goal_pixels = self.coords2pixels(goal, info, idx)
            cv2.circle(frontier_map_rgb, tuple(goal_pixels), 20, (0, 0, 255), -1)

            counter = 0
            for sub_goal, next_sub_goal in zip(
                self.sub_goals[idx][:-1], self.sub_goals[idx][1:]
            ):
                sub_goal = [sub_goal[2], sub_goal[0]]
                next_sub_goal = [next_sub_goal[2], next_sub_goal[0]]
                sub_goal_pixels = self.coords2pixels(sub_goal, info, idx)
                next_sub_goal_pixels = self.coords2pixels(next_sub_goal, info, idx)

                cv2.circle(
                    frontier_map_rgb,
                    tuple(next_sub_goal_pixels),
                    14,
                    (255, counter, counter),
                    -1,
                )
                cv2.line(
                    frontier_map_rgb,
                    tuple(sub_goal_pixels),
                    tuple(next_sub_goal_pixels),
                    (255, counter, counter),
                    4,
                )
                counter += 80

        frontier_map_rgb = cv2.resize(
            frontier_map_rgb,
            (
                int(frontier_map_rgb.shape[1] * downsize),
                int(frontier_map_rgb.shape[0] * downsize),
            ),
        )
        cv2.imshow("frontier_map", frontier_map_rgb)

        edge_map_rgb = self.edge_maps[idx]
        edge_map_rgb = cv2.resize(
            edge_map_rgb,
            (
                int(edge_map_rgb.shape[1] * downsize),
                int(edge_map_rgb.shape[0] * downsize),
            ),
        )
        cv2.imshow("edge_map", edge_map_rgb)

        cv2.imshow("rgb", obs["rgb"])
        cv2.waitKey(5)

    # select next frontier among frontier_goals based on some heuristic (greedy for now)
    def select_next_frontier(self, obs, frontier_goals, frontier_goals_info_gains):
        new_goal = frontier_goals[0]
        worst_info_gain = 0
        frontier_counter = 0

        # what is this?!
        while frontier_counter < len(frontier_goals) - 1:
            frontier_counter += 1
            n = np.random.randint(len(frontier_goals))
            new_goal = frontier_goals[n]
            new_info_gain = frontier_goals_info_gains[n]
            if new_info_gain < worst_info_gain:
                continue
            worst_info_gain = new_info_gain

        return new_goal

    # convert point in cv2 pixel coords to habitat 3D coords in (m)
    def pixels2coords(self, pixels, info, idx):
        lower_bound, upper_bound = self.envs.call_at(idx, "get_upper_and_lower_map_bounds")
        grid_size = (
            abs(upper_bound[2] - lower_bound[2]) / info["top_down_map"]["map"].shape[0],
            abs(upper_bound[0] - lower_bound[0]) / info["top_down_map"]["map"].shape[1],
        )
        # lower_bounds are inverted, why?!
        realworld_x = (
            lower_bound[0] + pixels[0] * grid_size[0]
        )  # goal in world cords (m)
        realworld_y = lower_bound[2] + pixels[1] * grid_size[1]

        return np.array([realworld_x, 0.0, realworld_y])

    # convert habitat 3D coords in (m) to cv2 pixel coords
    def coords2pixels(self, coords, info, idx):
        lower_bound, upper_bound = self.envs.call_at(idx, "get_upper_and_lower_map_bounds")
        grid_size = (
            abs(upper_bound[2] - lower_bound[2]) / info["top_down_map"]["map"].shape[0],
            abs(upper_bound[0] - lower_bound[0]) / info["top_down_map"]["map"].shape[1],
        )
        grid_x = int((coords[1] - lower_bound[0]) / grid_size[0])
        grid_y = int((coords[0] - lower_bound[2]) / grid_size[1])

        return [grid_x, grid_y]

    def generate(self) -> None:
        self.envs = self._init_train()
        generated_observations_paths = []

        while not self.is_done():
            if (
                self.num_steps_done % self.config.habitat.environment.max_episode_steps == 0
                or self.num_steps_done % self.config.habitat.environment.max_episode_steps > 20
            ):
                self._step(self.envs)

            for idx in range(self.envs.num_envs):
                obs = self.current_observations[idx]
                info = self.current_infos[idx]
                done = self.current_dones[idx]
                episode = self.envs.current_episodes()[idx]

                # self.envs.call_at(
                #     idx,
                #     "save_to_images",
                #     {"observations": obs, "info": info},
                # )

                # print("step:", self.num_steps_done, "done:", done, "episode:", episode.episode_id)
                if done:
                    self.replanning_time = [True] * self.envs.num_envs
                    self.sub_goals = [[] for _ in range(self.envs.num_envs)]
                    self.goal = [None] * self.envs.num_envs
                    self.previous_sub_goal = [None] * self.envs.num_envs
                    self.elapsed_since_last_sub_goal = [0] * self.envs.num_envs
                    self.got_new_plan = [False] * self.envs.num_envs
                    self.frontier_maps = [None] * self.envs.num_envs
                    # self.episode_elapsed_steps = [0] * self.envs.num_envs

                if self.visualize:
                    self.visualize_map(obs, info, idx)
                #n_step = self.envs.call_at(idx, "get_step")
                n_step = self.current_steps[idx]
                # save obs
                generated_observations_paths.append(
                    save_obs(self.exp_path, episode.episode_id, obs, n_step)
                )

                # for the first 10 steps, rotate in place
                if self.num_steps_done % self.config.habitat.environment.max_episode_steps < 21:
                    self.envs.call_at(
                        idx,
                        "step",
                        {"action": 2},
                    )
                    print("Rotating in place...")
                    self.num_steps_done += 1
                    continue

                self.elapsed_since_last_sub_goal[idx] += 1

                # check distance between agent and last subgoal
                previous_subgoal_reached = False
                if self.previous_sub_goal[idx] is not None:
                    distance_to_previous_subgoal = np.linalg.norm(
                        self.previous_sub_goal[idx] - obs["position"]["position"]
                    )
                    if distance_to_previous_subgoal < 0.25:
                        previous_subgoal_reached = True

                        # TODO: check if last frontier has been reached, then replan
                        if len(self.sub_goals[idx]) == 0:
                            self.replanning_time[idx] = True
                            self.previous_sub_goal[idx] = None

                # check if time to go to next subgoal (either a new goal is available, or 20 steps elapsed)
                if len(self.sub_goals[idx]) > 0 and (
                    previous_subgoal_reached
                    or self.got_new_plan[idx]
                    or self.elapsed_since_last_sub_goal[idx] > 40
                ):
                    new_sub_goal = self.sub_goals[idx].pop(0)
                    self.got_new_plan[idx] = False
                    print(
                        "picking next subgoal. Remaining subgoals=",
                        len(self.sub_goals[idx]),
                    )
                    self.envs.call_at(
                        idx,
                        "set_goals",
                        {"data": [NavigationGoal(position=new_sub_goal)]},
                    )
                    self.agent.sub_goals[idx] = NavigationGoal(position=new_sub_goal)
                    self.previous_sub_goal[idx] = new_sub_goal
                    self.elapsed_since_last_sub_goal[idx] = 0

                # if it's time to replan
                if self.replanning_time[idx]:
                    self.replanning_time[idx] = False

                    frontier_goals, frontier_goals_info_gains = self.compute_frontiers(
                        obs, info, idx
                    )

                    # greedily select next exploration goal among frontier goals and compute subgoals using pathfinder
                    if len(frontier_goals) > 0:
                        agent_position = obs["position"]["position"]

                        new_goal_pixels = self.select_next_frontier(
                            obs, frontier_goals, frontier_goals_info_gains
                        )
                        new_goal_coords = self.pixels2coords(new_goal_pixels, info, idx)
                        new_goal_coords[1] = agent_position[1]

                        self.goal[idx] = new_goal_coords

                        print(
                            "New frontier goal at:", new_goal_coords, "- sending agent"
                        )

                        path_points = self.envs.call_at(
                            idx,
                            "get_path",
                            {"agent_position": agent_position, "goal": new_goal_coords},
                        )
                        # print("Path:", path_points)
                        self.sub_goals[idx] = path_points.tolist()
                        self.got_new_plan[idx] = (
                            True  # signal that a new plan is available
                        )

                    else:  # if no new frontiers found, map is fully explored -> reset episode
                        print("No new frontiers, sending agent back to origin (0, 0)")
                        self.envs.call_at(
                            idx,
                            "set_goals",
                            {
                                "data": [
                                    NavigationGoal(
                                        position=[
                                            0,
                                            obs["position"]["position"][1],
                                            0,
                                        ]
                                    )
                                ]
                            },
                        )
                        self.got_new_plan[idx] = False

                self.num_steps_done += 1

        self.envs.close()

        return sorted(generated_observations_paths)
