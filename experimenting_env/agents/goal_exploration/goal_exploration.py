# type: ignore
import math
import copy
import os
import time
import numpy as np
from collections import deque
from typing import *
import random
import gym
import open_clip
import torch
import wandb
from detectron2.structures import Boxes, Instances
from habitat.tasks.nav.nav import NavigationGoal, NavigationTask
from habitat.utils.visualizations import maps
from habitat_baselines.agents.simple_agents import GoalFollower
from habitat_baselines.common.baseline_registry import baseline_registry
import string
from experimenting_env.captioner.utils.utils import Configuration
from experimenting_env.captioner.models.coca.coca import CoCa
from experimenting_env.captioner.models.blip2.blip2 import BLIP2
from experimenting_env.agents.baselines import SubGoalFollower
from experimenting_env.agents.model import *
from experimenting_env.agents.ppo import *
from experimenting_env.detector.model import MultiStageModel
from experimenting_env.utils.sensors_utils import save_obs
from experimenting_env.utils.skeleton import *
from experimenting_env.utils.storage import *
from experimenting_env.utils.matching import get_objects_ids
from experimenting_env.utils import projection_utils as pu
from PIL import Image
from sentence_transformers import SentenceTransformer

from ..baselines import Baseline


@baseline_registry.register_trainer(name="goalexplorationbaseline-v0")
class GoalExplorationBaseline(Baseline):

    def __init__(self, config, exp_base_dir, detectron_args, **kwargs):
        super().__init__(config, exp_base_dir, SubGoalFollower, **kwargs)
        self.config = config
        self.device_config = self.config.device_config

        # reduced canonical map size for network input
        self.map_width, self.map_height = 128, 128
        self.visualize = self.config.ppo.visualize  # visualize maps

        self.object_detector = ObjectDetectorEnv({"cfg": detectron_args}, device=self.device_config.object_detector.device)
        #self.object_detector_gt = ObjectDetectorGTEnv()
        
        self.captioner = self.get_captioner(self.config.captioner)

        self.captioner.to(self.device_config.captioner.device)
        self.captioner.eval()
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2").to(self.device_config.sentence_transformer.device)

    def _step(self, envs):
        super()._step(envs)
        self.predict_current_bbs_and_update_pcd()

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

    def predict_current_bbs_and_update_pcd(self, detector_batch_size=8):
        # predict bbs with object detector

        for idx in range(0, self.envs.num_envs, detector_batch_size):
            max_idx = min(self.envs.num_envs, idx + detector_batch_size)
            images = [self.current_observations[i]["rgb"] for i in range(idx, max_idx)]
            preds = self.object_detector.predict_batch(images)

            for i in range(max_idx - idx):
                current_env = idx + i
                embeddings = [] 
                captions = []
                gt_logits_list = []

                # self.current_infos[current_env]['top_down_map'] = self.envs.call_at(current_env, "get_topdown_map")
                #objects_annotations = self.envs.call_at(0, "get_semantic_annotations")
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

    def get_rewards(self, current_observations):
        return torch.from_numpy(
            np.array(
                [
                    self.envs.call_at(idx, "get_reward", {'disagreement_map': current_observations[idx]["disagreement_map"]})
                    for idx in range(self.envs.num_envs)
                ]
            )
        )

    def create_policy_inputs(self):
        # create disagreeemt inputs
        disagreement_inputs = [
            self.current_observations[idx]["disagreement_map"]
            for idx in range(self.envs.num_envs)
        ]
        disagreement_inputs = [
            cv2.resize(x, (self.map_height, self.map_width)).reshape(
                1, 1, self.map_height, self.map_width
            )
            for x in disagreement_inputs
        ]
        disagreement_inputs = torch.tensor(disagreement_inputs).reshape(
            self.envs.num_envs, 1, self.map_height, self.map_width
        )

        # create position inputs
        position_inputs = []
        for idx in range(self.envs.num_envs):
            agent_pos = [
                self.current_observations[idx]["position"]["position"][2],
                self.current_observations[idx]["position"]["position"][0],
            ]

            position_input = cv2.resize(
                self.current_infos[idx]["top_down_map"]["map"],
                (
                    self.current_observations[idx]["disagreement_map"].shape[1],
                    self.current_observations[idx]["disagreement_map"].shape[0],
                ),
            )

            lower_bound, upper_bound = self.envs.call_at(idx, "get_upper_and_lower_map_bounds")

            grid_size = (
                abs(upper_bound[2] - lower_bound[2]) / position_input.shape[0],
                abs(upper_bound[0] - lower_bound[0]) / position_input.shape[1],
            )

            grid_x = int((agent_pos[1] - lower_bound[0]) / grid_size[0])
            grid_y = int((agent_pos[0] - lower_bound[2]) / grid_size[1])
            cv2.circle(position_input, (grid_x, grid_y), 10, 3, -1)

            position_input = cv2.resize(
                position_input, (self.map_height, self.map_width)
            ).reshape(1, 1, self.map_height, self.map_width)
            position_inputs.append(position_input)

        position_inputs = torch.tensor(position_inputs).reshape(
            self.envs.num_envs, 1, self.map_height, self.map_width
        )
        # concat inputs

        global_inputs = torch.cat((disagreement_inputs, position_inputs), dim=1)

        return global_inputs

    def goto_next_subgoal(self, idx):
        if (self.got_new_plan[idx] or self.current_steps[idx] % 20 == 0) and len(
            self.sub_goals[idx]
        ) > 0:
            new_sub_goal = self.sub_goals[idx].pop(-1)
            self.agent.sub_goals[idx] = NavigationGoal(position=new_sub_goal)
            self.got_new_plan[idx] = False
            print("Picking next subgoal. Remaining subgoals=", len(self.sub_goals[idx]))

    def predict_new_goals_batched(self):
        if (
            self.first_step
            or self.current_steps[0] % self.config.ppo.replanning_steps == 0
        ):
            print("Predicting new goals...")
            self.first_step = False

            # add new samples to global policy storage
            self.g_rollouts.insert(
                self.global_input,
                self.g_rec_states,
                self.g_action,
                self.g_action_log_prob,
                self.g_value,
                self.g_reward,
                self.g_masks,
                self.global_orientation,
            )

            # sample next goal
            (
                self.g_value,
                self.g_action,
                self.g_action_log_prob,
                self.g_rec_states,
            ) = self.g_policy.act(
                inputs=self.g_rollouts.obs[-1],
                rnn_hxs=self.g_rollouts.rec_states[-1],
                masks=self.g_rollouts.masks[-1],
                extras=self.g_rollouts.extras[-1],
                deterministic=False,
            )

            cpu_actions = nn.Sigmoid()(self.g_action).cpu().numpy()

            # get the current map
            mymaps = [
                self.current_infos[idx]["top_down_map"]['map'].copy() * 255
                for idx in range(self.envs.num_envs)
            ]

            mymaps_sizes_pixels = [x.shape for x in mymaps]

            for action in cpu_actions:
                print("goal:", action)
            # rescaled_pixel_goals = [[int(action[0] * mymaps_sizes_pixels[i][1]), int(action[1] * mymaps_sizes_pixels[i][0])]
            #                 for i, action in enumerate(cpu_actions)]
            rescaled_pixel_goals = [
                [
                    int(
                        action[0]
                        * self.current_observations[i]['disagreement_map'].shape[1]
                    ),
                    int(
                        action[1]
                        * self.current_observations[i]['disagreement_map'].shape[0]
                    ),
                ]
                for i, action in enumerate(cpu_actions)
            ]
            # compute subgoals for each goal
            for idx, rescaled_pixel_goal in enumerate(rescaled_pixel_goals):
                # convert pixels to global map coords
                lower_bound, upper_bound = self.envs.call_at(idx, "get_upper_and_lower_map_bounds")
                grid_resolution = (
                    abs(upper_bound[2] - lower_bound[2]) / mymaps_sizes_pixels[idx][0],
                    abs(upper_bound[0] - lower_bound[0]) / mymaps_sizes_pixels[idx][1],
                )

                # lower_bounds are inverted, why?!
                realworld_x = (
                    lower_bound[0] + rescaled_pixel_goal[0] * grid_resolution[0]
                )
                realworld_y = (
                    lower_bound[2] + rescaled_pixel_goal[1] * grid_resolution[1]
                )

                goal_world = [
                    realworld_x,
                    realworld_y,
                ]  # goal in world coords (m)

                agent_pos = [
                    self.current_observations[idx]['position']['position'][2],
                    self.current_observations[idx]['position']['position'][0],
                ]

                scaled_down_map = cv2.resize(
                    mymaps[idx],
                    (
                        self.current_observations[idx]['disagreement_map'].shape[1],
                        self.current_observations[idx]['disagreement_map'].shape[0],
                    ),
                )

                grid_size = (
                    abs(upper_bound[2] - lower_bound[2]) / scaled_down_map.shape[0],
                    abs(upper_bound[0] - lower_bound[0]) / scaled_down_map.shape[1],
                )

                grid_x = int((agent_pos[1] - lower_bound[0]) / grid_size[0])
                grid_y = int((agent_pos[0] - lower_bound[2]) / grid_size[1])

                # set start end end pose for A*
                start_pos = [grid_x, grid_y]
                end_pos = [
                    int(rescaled_pixel_goals[0][0]),
                    int(rescaled_pixel_goals[0][1]),
                ]

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
                    self.astar_img[idx] = cv2.resize(
                        self.astar_img[idx],
                        (
                            self.current_observations[idx]['disagreement_map'].shape[
                                1
                            ],
                            self.current_observations[idx]['disagreement_map'].shape[
                                0
                            ],
                        ),
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
                        self.current_observations[idx]['position']['position'][1],
                        realworld_subgoal_y,
                    ]

                    self.sub_goals[idx].append(realworld_subgoal)

                self.sub_goals[idx].pop(-1)

                if len(self.sub_goals[idx]) > 0:
                    self.got_new_plan[idx] = True
                else:
                    print("A* failed, no waypoints")
                    
    def predict_new_goals(self):
        if (
            self.first_step
            or self.current_steps[0] % self.config.ppo.replanning_steps == 0
        ):
            self.add_sample_to_gpolicy()
            for env_idx in range(self.envs.num_envs):
                if (
                    self.first_step
                    or self.current_steps[0] % self.config.ppo.replanning_steps == 0
                ):
                    self.compute_new_goals(env_idx)

            self.first_step = False

    def add_sample_to_gpolicy(self):
        print("Predicting new goals...")

        # add new samples to global policy storage
        self.g_rollouts.insert(
            self.global_input,
            self.g_rec_states,
            self.g_action,
            self.g_action_log_prob,
            self.g_value,
            self.g_reward,
            self.g_masks,
            self.global_orientation,
        )
    def compute_new_goals(self, env_idx):

        tentative_counter = 0 
        path_founded = False

        while tentative_counter < self.max_goal_tentative and not path_founded:
        
            # sample next goal
            (
                self.g_value,
                self.g_action,
                self.g_action_log_prob,
                self.g_rec_states,
            ) = self.g_policy.act(
                inputs=self.g_rollouts.obs[-1][env_idx:env_idx + 1],
                rnn_hxs=self.g_rollouts.rec_states[-1][env_idx:env_idx + 1],
                masks=self.g_rollouts.masks[-1][env_idx:env_idx + 1],
                extras=self.g_rollouts.extras[-1][env_idx:env_idx + 1],
                deterministic=False,
            )

            cpu_actions = nn.Sigmoid()(self.g_action).cpu().numpy()

            # get the current map
            mymaps = self.current_infos[env_idx]["top_down_map"]["map"].copy() * 255

            mymaps_sizes_pixels = mymaps.shape 

            for action in cpu_actions:
                print("goal:", action)
            # rescaled_pixel_goals = [[int(action[0] * mymaps_sizes_pixels[i][1]), int(action[1] * mymaps_sizes_pixels[i][0])]
            #                 for i, action in enumerate(cpu_actions)]
            rescaled_pixel_goal = [
                int(
                    action[0]
                    * self.current_observations[env_idx]["disagreement_map"].shape[1]
                ),
                int(
                    action[1]
                    * self.current_observations[env_idx]["disagreement_map"].shape[0]
                ),
            ]
            
            # convert pixels to global map coords
            lower_bound, upper_bound = self.envs.call_at(env_idx, "get_upper_and_lower_map_bounds")
            grid_resolution = (
                abs(upper_bound[2] - lower_bound[2]) / mymaps_sizes_pixels[0],
                abs(upper_bound[0] - lower_bound[0]) / mymaps_sizes_pixels[1],
            )

            # lower_bounds are inverted, why?!
            realworld_x = (
                lower_bound[0] + rescaled_pixel_goal[0] * grid_resolution[0]
            )
            realworld_y = (
                lower_bound[2] + rescaled_pixel_goal[1] * grid_resolution[1]
            )

            goal_world = [
                realworld_x,
                realworld_y,
            ]  # goal in world coords (m)

            agent_pos = [
                self.current_observations[env_idx]["position"]["position"][2],
                self.current_observations[env_idx]["position"]["position"][0],
            ]

            scaled_down_map = cv2.resize(
                mymaps,
                (
                    self.current_observations[env_idx]["disagreement_map"].shape[1],
                    self.current_observations[env_idx]["disagreement_map"].shape[0],
                ),
            )

            grid_size = (
                abs(upper_bound[2] - lower_bound[2]) / scaled_down_map.shape[0],
                abs(upper_bound[0] - lower_bound[0]) / scaled_down_map.shape[1],
            )

            grid_x = int((agent_pos[1] - lower_bound[0]) / grid_size[0])
            grid_y = int((agent_pos[0] - lower_bound[2]) / grid_size[1])

            # set start end end pose for A*
            start_pos = [grid_x, grid_y]
            end_pos = [
                int(rescaled_pixel_goal[0]),
                int(rescaled_pixel_goal[1]),
            ]

            end_pos[0] = max(0, min(end_pos[0], scaled_down_map.shape[1] - 1))
            end_pos[1] = max(0, min(end_pos[1], scaled_down_map.shape[0] - 1))

            # extract a path to the goal (path is in opencv pixel coords)
            path, img = do_plan(
                scaled_down_map, start_pos, end_pos, False, True, end_pos
            )
            if len(path) == 0:
                print("No path!")
                self.sub_goals[env_idx] = []
                print(f"Calculating new goal for env {env_idx} -- Tentative {tentative_counter}")
                tentative_counter += 1
                continue
            print(f"Found path for env {env_idx}")
            path.insert(0, end_pos)
            path_founded = True

            if self.visualize:
                self.astar_img[env_idx] = img.copy()
                cv2.circle(
                    self.astar_img[env_idx],
                    (end_pos[0], end_pos[1]),
                    20,
                    (255, 255, 0),
                    4,
                )
                self.astar_img[env_idx] = cv2.resize(
                    self.astar_img[env_idx],
                    (
                        self.current_observations[env_idx]["disagreement_map"].shape[
                            1
                        ],
                        self.current_observations[env_idx]["disagreement_map"].shape[
                            0
                        ],
                    ),
                )

            # extract sub-goals
            self.sub_goals[env_idx] = []
            for subgoal in path:
                realworld_subgoal_x = lower_bound[0] + subgoal[0] * (
                    abs(upper_bound[0] - lower_bound[0]) / scaled_down_map.shape[1]
                )
                realworld_subgoal_y = lower_bound[2] + subgoal[1] * (
                    abs(upper_bound[2] - lower_bound[2]) / scaled_down_map.shape[0]
                )
                realworld_subgoal = [
                    realworld_subgoal_x,
                    self.current_observations[env_idx]["position"]["position"][1],
                    realworld_subgoal_y,
                ]

                self.sub_goals[env_idx].append(realworld_subgoal)

            self.sub_goals[env_idx].pop(-1)

            if len(self.sub_goals[env_idx]) > 0:
                self.got_new_plan[env_idx] = True
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

        self.first_step = True
        self.max_goal_tentative = self.config.habitat_baselines.goal_tentative

        # goal policy observation space
        self.g_observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(2, self.map_width, self.map_height), dtype="uint8"
        )

        # goal policy action space
        self.g_action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # goal policy init
        device = torch.device("cuda:0" if self.config.ppo.cuda else "cpu")

        self.g_policy = RL_Policy(
            self.g_observation_space.shape,
            self.g_action_space,
            base_kwargs={
                "recurrent": self.config.ppo.use_recurrent_global,
                "hidden_size": self.config.ppo.g_hidden_size,
                "downscaling": self.config.ppo.global_downscaling,
            },
        ).to(device)

        # rollout storage
        self.g_rollouts = GlobalRolloutStorage(
            self.config.ppo.num_global_steps,
            self.envs.num_envs,
            self.g_observation_space.shape,
            self.g_action_space,
            self.g_policy.rec_state_size,
            1,
        ).to(device)

        # create queues
        self.g_value_losses = deque(maxlen=1000)
        self.g_action_losses = deque(maxlen=1000)
        self.g_dist_entropies = deque(maxlen=1000)

        # first forward pass

        self.global_input = torch.rand(
            self.envs.num_envs, 2, self.map_height, self.map_width
        )
        self.global_orientation = torch.rand(self.envs.num_envs, 1)

        self.g_rollouts.obs[0].copy_(self.global_input)
        self.g_masks = torch.ones(self.envs.num_envs).float().to(device)  # not used
        self.g_rollouts.extras[0].copy_(self.global_orientation)

        torch.set_grad_enabled(False)

        # run goal policy (predict global goals)
        (
            self.g_value,
            self.g_action,
            self.g_action_log_prob,
            self.g_rec_states,
        ) = self.g_policy.act(
            inputs=self.g_rollouts.obs[0],
            rnn_hxs=self.g_rollouts.rec_states[0],
            masks=self.g_rollouts.masks[0],
            extras=self.g_rollouts.extras[0],
            deterministic=False,
        )
        if self.config.ppo.load_checkpoint:
            if os.path.exists(self.config.ppo.load_checkpoint_path):
                self.g_policy.load_state_dict(
                    torch.load(self.config.ppo.load_checkpoint_path)
                )
            else:
                print("Error: checkpoint path does not exist.")
                return

        print("GoalExplorationBaseline init done...")

        # data generation loop
        while not self.is_done():
            # step all envs
            self._step(self.envs)

            not_ready = False
            # if no observations or no disagr map, exit
            if not self.current_infos or not self.current_observations:
                print("\n\n\nINFOS OR OBS NOT READY")
                not_ready = True

            for idx in range(self.envs.num_envs):
                if "disagreement_map" not in self.current_observations[idx]:
                    print("\n\n\nDISAGREEMENT NOT READY FOR IDX:", idx)
                    not_ready = True

            if not_ready:
                continue

            # for each env, visualize and send next subgoal if needed
            for idx in range(self.envs.num_envs):
                obs = self.current_observations[idx]
                episode = self.envs.current_episodes()[idx]
                info = self.current_infos[idx]
                episode = self.envs.current_episodes()[idx]
                #TODO: Check if self.current_steps is correct
                # n_step = self.envs.call_at(idx, "get_step")
                n_step = self.current_steps[idx]
                paths = save_obs(self.exp_path, episode.episode_id, obs, n_step)
                generated_observations_paths.append(paths)

                # visualize maps for debugging purposes
                if self.visualize and self.astar_img[idx] is not None:
                    cv2.imshow("Env " + str(idx) + " A*", self.astar_img[idx])
                if self.visualize and ("disagreement_map" in obs):
                    rgb_map = maps.colorize_draw_agent_and_fit_to_height(
                        info["top_down_map"], obs["disagreement_map"].shape[0]
                    )
                    cv2.imshow("Env " + str(idx) + " map", rgb_map)
                    cv2.imshow(
                        "Env " + str(idx) + " disagreement", obs["disagreement_map"]
                    )

                im = obs['rgb'].copy()
                if self.visualize and len(self.current_observations[idx]['bbs']['instances'].pred_boxes)!= 0: 
                    for j in range(len(self.current_observations[idx]['bbs']['instances'].pred_boxes)): 
                        x, y, w, h = self.current_observations[idx]['bbs']['instances'].pred_boxes[:].tensor[j]
                        x, y, w, h = int(x.item()), int(y.item()), int(w.item()), int(h.item())
                        cv2.rectangle(im, (x, y), (w, h), (0, 255, 0))
                        cv2.putText(im, self.current_observations[idx]['bbs']['instances'].captions[j], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 2)

                if self.visualize:
                    cv2.imshow("rgb", im)
                    cv2.waitKey(5)

                # if time to go to next subgoal, do it
                self.goto_next_subgoal(idx)

            if self.visualize:
                cv2.waitKey(10)

            # get reward
            self.g_reward = self.get_rewards(self.current_observations)

            # populate inputs
            # map, position input
            self.global_input = self.create_policy_inputs()

            # orientation input
            agent_orient = [
                self.current_observations[idx]["position"]["orientation"]
                for idx in range(self.envs.num_envs)
            ]
            try:
                agent_orientation = [
                    2.0 * math.acos(np.clip(q.y, -1, 1)) for q in agent_orient
                ]
            except:
                breakpoint()
            self.global_orientation = torch.tensor(agent_orientation).reshape(
                self.envs.num_envs, 1
            )

            # if time to predict new goal
            self.predict_new_goals()

            for idx in range(self.envs.num_envs):
                done = self.current_dones[idx]

                self.num_steps_done += 1
                if done:
                    self.current_steps[idx] = 0

            if self.num_steps_done % 10 == 0:
                print(f"Progress: {int(self.percent_done() * 100)}%")
        del self.current_dones
        del self.current_observations

        self.envs.close()
        return sorted(generated_observations_paths)

    def _training_log(
        self,
        reward: float,
        losses: Dict[str, float],
        metrics: Dict[str, float] = None,
        prev_time: int = 0,
    ):
        wandb.log(
            {"reward": reward, "num_steps_done": self.num_steps_done, "losses": losses}
        )

        if metrics:
            wandb.log(
                {
                    "metrics": metrics,
                    "num_steps_done": self.num_steps_done,
                }
            )

    def train(self) -> None:
        self.envs = self._init_train()
        self.sub_goals = [[] for _ in range(self.envs.num_envs)]
        self.sub_goals_counter = [0] * self.envs.num_envs
        self.got_new_plan = [False] * self.envs.num_envs
        self.replan_retries = [0] * self.envs.num_envs
        self.replan = [True] * self.envs.num_envs
        self.astar_img = [None] * self.envs.num_envs

        self.first_step = True
        self.max_goal_tentative = self.config.habitat_baselines.goal_tentative

        # goal policy observation space
        self.g_observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(2, self.map_width, self.map_height), dtype="uint8"
        )

        # goal policy action space
        self.g_action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # goal policy init
        device = torch.device("cuda:0" if self.config.ppo.cuda else "cpu")

        self.g_policy = RL_Policy(
            self.g_observation_space.shape,
            self.g_action_space,
            base_kwargs={
                "recurrent": self.config.ppo.use_recurrent_global,
                "hidden_size": self.config.ppo.g_hidden_size,
                "downscaling": self.config.ppo.global_downscaling,
            },
        ).to(device)

        # ppo agent init

        self.g_agent = PPO(
            self.g_policy,
            self.config.ppo.clip_param,
            self.config.ppo.ppo_epoch,
            self.envs.num_envs,
            self.config.ppo.value_loss_coeff,
            self.config.ppo.entropy_coef,
            lr=self.config.ppo.global_lr,
            eps=self.config.ppo.eps,
            max_grad_norm=self.config.ppo.max_grad_norm,
        )

        # rollout storage
        self.g_rollouts = GlobalRolloutStorage(
            self.config.ppo.num_global_steps,
            self.envs.num_envs,
            self.g_observation_space.shape,
            self.g_action_space,
            self.g_policy.rec_state_size,
            1,
        ).to(device)

        # create queues
        self.g_value_losses = deque(maxlen=1000)
        self.g_action_losses = deque(maxlen=1000)
        self.g_dist_entropies = deque(maxlen=1000)

        # first forward pass
        self.global_input = torch.rand(
            self.envs.num_envs, 2, self.map_height, self.map_width
        )
        self.global_orientation = torch.rand(self.envs.num_envs, 1)

        self.g_rollouts.obs.copy_(self.global_input)
        self.g_masks = torch.ones(self.envs.num_envs).float().to(device)  # not used
        self.g_rollouts.extras[0].copy_(self.global_orientation)

        torch.set_grad_enabled(False)

        # run goal policy (predict global goals)
        (
            self.g_value,
            self.g_action,
            self.g_action_log_prob,
            self.g_rec_states,
        ) = self.g_policy.act(
            inputs=self.g_rollouts.obs[0],
            rnn_hxs=self.g_rollouts.rec_states[0],
            masks=self.g_rollouts.masks[0],
            extras=self.g_rollouts.extras[0],
            deterministic=False,
        )

        self.t_start = time.time()
        self.checkpoint_folder = self.config.habitat_baselines.checkpoint_folder

        if not os.path.exists(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)

        if self.config.ppo.load_checkpoint:
            if os.path.exists(self.config.ppo.load_checkpoint_path):
                self.g_policy.load_state_dict(
                    torch.load(self.config.ppo.load_checkpoint_path)
                )
            else:
                print("Error: checkpoint path does not exist.")
                return

        # training loop
        while not self.is_done():
            # step all envs
            self._step(self.envs)

            not_ready = False
            # if no observations or no disagr map, exit
            if not self.current_infos or not self.current_observations:
                print("\n\n\nINFOS OR OBS NOT READY")
                not_ready = True

            for idx in range(self.envs.num_envs):
                if "disagreement_map" not in self.current_observations[idx]:
                    print("\n\n\nDISAGREEMENT NOT READY FOR IDX:", idx)
                    not_ready = True

            if not_ready:
                continue

            # for each env, visualize and send next subgoal if needed
            for idx in range(self.envs.num_envs):
                obs = self.current_observations[idx]
                info = self.current_infos[idx]
                episode = self.envs.current_episodes()[idx]
                n_step = self.current_steps[idx]
                
                # visualize maps for debugging purposes
                if self.visualize and self.astar_img[idx] is not None:
                    img = wandb.Image(self.astar_img[idx])
                    wandb.log({f"env_{str(idx)}_a*": img})
                if self.visualize and ("disagreement_map" in obs):
                    rgb_map = maps.colorize_draw_agent_and_fit_to_height(
                        info["top_down_map"], obs["disagreement_map"].shape[0]
                    )
                    img = wandb.Image(rgb_map)
                    wandb.log({f"env_{str(idx)}_map": img})

                    img = wandb.Image(obs["disagreement_map"])
                    wandb.log({f"env_{str(idx)}_disagreement_map": img})

                # is it time to go to next subgoal?
                self.goto_next_subgoal(idx)

            # get reward
            self.g_reward = self.get_rewards(self.current_observations)

            # populate inputs
            # map, position input
            self.global_input = self.create_policy_inputs()

            # orientation input
            agent_orient = [
                self.current_observations[idx]["position"]["orientation"]
                for idx in range(self.envs.num_envs)
            ]
            agent_orientation = [2.0 * math.acos(q.y) for q in agent_orient]
            self.global_orientation = torch.tensor(agent_orientation).reshape(
                self.envs.num_envs, 1
            )
            self.predict_new_goals_batched()
                        
            # train goal policy
            if self.num_steps_done % self.config.ppo.num_global_steps == 0:
                torch.set_grad_enabled(True)

                self.g_next_value = self.g_policy.get_value(
                    self.g_rollouts.obs[-1],
                    self.g_rollouts.rec_states[-1],
                    self.g_rollouts.masks[-1],
                    extras=self.g_rollouts.extras[-1],
                ).detach()

                self.g_rollouts.compute_returns(
                    self.g_next_value,
                    self.config.ppo.use_gae,
                    self.config.ppo.gamma,
                    self.config.ppo.tau,
                )
                (
                    self.g_value_loss,
                    self.g_action_loss,
                    self.g_dist_entropy,
                ) = self.g_agent.update(self.g_rollouts)

                self.g_value_losses.append(self.g_value_loss)
                self.g_action_losses.append(self.g_action_loss)
                self.g_dist_entropies.append(self.g_dist_entropy)
                self.g_rollouts.after_update()

                torch.set_grad_enabled(False)

                losses = {
                    "value_loss": self.g_value_loss,
                    "action_loss": self.g_action_loss,
                }

                reward = self.g_reward.mean()
                self._training_log(reward, losses)

            # save measures
            # if (
            #     int(self.current_steps[0])
            #     % (self.config.habitat.environment.max_episode_steps - 2)
            #     == 0
            # ):
            # metrics = {
            #     "episode_reward": self.current_infos[0]["episode_reward"],
            #     "area_ratio": self.current_infos[0]["area_ratio"],
            # }
            # wandb.log({"metrics": metrics, "num_steps_done": self.num_steps_done})

            if self.num_steps_done % self.config.ppo.save_periodic == 0:
                torch.save(
                    self.g_policy.state_dict(),
                    os.path.join(
                        self.checkpoint_folder, f"checkpoint_{self.num_steps_done}.ckpt"
                    ),
                )

            self.num_steps_done += 1
            print("Steps done:", self.num_steps_done)

        self.envs.close()


class ObjectDetectorGTEnv:
    MATTERPORT_SIM_TO_COCO_MAPPING = {
        5:60,
        3: 56,  # chair
        10: 57,  # couch
        14: 58,  # plan
        11: 59,  # beed
        18: 61,  # toilet
        22: 62,  # tv

    }

    SIM_TO_COCO_MAPPING = {
        #"chair": 56,  # chair
        "couch": 57,  # couch
        "potted plant": 58,  # plan
        "bed": 59,  # bed
        "toilet": 61,  # toilet
        "tv": 62,  # tv
        "dining table": 60,  # dining table
    }

    DEPTH_THR = 5

    def __init__(self):
        self.scene = ""
        self.filter_occluded_instances = True

    def convert_matterport_to_coco_labels(self, label):
        switch = {
            'table': 'dining table',
            'plant': 'potted plant',
            'sofa': 'couch',
            'tv_monitor': 'tv',
        }
        if label in switch.keys():
            return switch[label]
        else:
            return label

    def get_observation(self, sense, mapping, scene):
        # sense = kwargs['observations']['semantic']

        current_scene = scene
        if current_scene != self.scene:

            self.mapping = mapping
            self.scene = current_scene
            print("Updating mapping")

        bounding_boxes = []
        classes = []
        pred_masks = []
        infos = []
        mask_cleaned = sense.astype('uint8')

        objects_id = np.unique(mask_cleaned)
        for id_object in objects_id:

            bb = cv2.boundingRect((mask_cleaned == id_object).astype('uint8'))
            x, y, w, h = bb
            if id_object not in self.mapping:
                continue

            if (mask_cleaned == id_object).sum() < 1000:
                continue
            habitat_id = self.convert_matterport_to_coco_labels(
                self.mapping[id_object]
            )

            if habitat_id in self.SIM_TO_COCO_MAPPING:
                pred_mask = torch.zeros(sense.shape, dtype=torch.bool)
                pred_mask[(mask_cleaned == id_object).astype('bool')] = True

                coco_id = self.SIM_TO_COCO_MAPPING[habitat_id]
                bounding_boxes.append(torch.tensor([x, y, x + w, y + h]).unsqueeze(0))
                classes.append(torch.tensor(coco_id).unsqueeze(0))
                pred_masks.append(pred_mask.unsqueeze(0))
                infos.append(
                    {
                        'id_object': id_object,
                        # 'center': self._objects[id_object].aabb.center,
                    }
                )

        if len(bounding_boxes) > 0:
            results = Instances(
                image_size=sense.shape,
                pred_boxes=Boxes(torch.cat(bounding_boxes)),
                pred_classes=torch.cat(classes),
                scores=torch.ones(len(bounding_boxes)),
                pred_masks=torch.cat(pred_masks),
                infos=np.array(infos),
            )

        else:
            results = Instances(
                pred_boxes=Boxes(torch.Tensor()),
                image_size=sense.shape,
                pred_classes=torch.Tensor(),
                pred_masks=torch.Tensor(),
                scores=torch.Tensor(),
                infos=infos,
            )

        return {'instances': results}

class ObjectDetectorEnv:
    def __init__(self, config, device="cpu"):
        self.object_detector = MultiStageModel(**config)
        self.device = device

        self.object_detector.to(device)
        self.object_detector.eval()
        #self.object_detector.model.roi_heads.box_predictor.box_predictor.test_score_thresh = 0.5

    def load(self, load_path):
        if os.path.exists(load_path):
            self.object_detector.load_state_dict(torch.load(load_path)).to(self.device)

    def move_to_device(self, device="cpu"):
        self.device = device
        self.object_detector.to(device)

    def predict_batch(self, images):
        predictions = self.object_detector.infer(images)
        for i in range(len(images)):
            predictions[i]["instances"] = predictions[i]["instances"].to("cpu")

        return predictions
