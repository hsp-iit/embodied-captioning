import os
import numpy as np
import cv2
import gc

import torch

from experimenting_env.agents.baselines import Baseline
from experimenting_env.captioner.utils.utils_detector import select_object_detector
from experimenting_env.utils.skeleton import do_plan
from habitat.utils.visualizations import maps
from experimenting_env.captioner.utils.utils_captioner import select_captioner
from habitat_baselines.agents.simple_agents import GoalFollower
# from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from experimenting_env.utils.habitat_utils import construct_envs
from experimenting_env.utils.sensors_utils import save_obs
from habitat.tasks.nav.nav import NavigationGoal

from third_parties.detectron2.detectron2.evaluation import print_csv_format


class BaselineRLCaptioner(Baseline):
    def __init__(self, config, exp_base_dir, agent_class, **kwargs):
        super().__init__(config, exp_base_dir, agent_class, **kwargs)

        self.config = config
        self.kwargs = kwargs

        self.agent = agent_class(
            self.config.habitat_baselines.success_distance,
            self.config.habitat_baselines.goal_sensor_uuid,
        )

        self.exp_path = os.path.join(os.getcwd(), "dataset")
        self.captioner = select_captioner(config.captioner)
        self.captioner.eval()
        self.object_detector_gt = select_object_detector(config.detector)

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
            act = self.agent.act(self.current_observations[index_env])
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


@baseline_registry.register_trainer(name="randomgoalsbaselinecaptioner")
class RandomGoalsBaselineCaptioner(BaselineRLCaptioner):
    def __init__(self, config, exp_base_dir, **kwargs):
        super().__init__(config, exp_base_dir, GoalFollower, **kwargs)
        self.config = config

        self.visualize = self.config.randomgoals.visualize  # visualize maps

    def goto_next_subgoal(self, idx):
        if (self.got_new_plan[idx] or self.current_steps[idx] % 10 == 0) and len(
                self.sub_goals[idx]
        ) > 0:
            new_sub_goal = self.sub_goals[idx].pop(-1)
            self.got_new_plan[idx] = False
            print("Picking next subgoal. Remaining subgoals=", len(self.sub_goals[idx]))
            self.envs.call_at(
                idx, "set_goals", {"data": [NavigationGoal(position=new_sub_goal)]}
            )

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
                lower_bound, upper_bound = self.envs.call_at(idx,
                                                             "get_upper_and_lower_map_bounds")  # "get_map_bounds" in habitat2

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
                    # self.current_observations[idx]["position"]["position"][2],
                    # self.current_observations[idx]["position"]["position"][0],
                    self.current_observations[idx]["gps"][1],
                    self.current_observations[idx]["gps"][0],
                ]

                scaled_down_map = mymaps[idx]  # no rescalement required

                grid_size = (
                    abs(upper_bound[2] - lower_bound[2]) / scaled_down_map.shape[0],
                    abs(upper_bound[0] - lower_bound[0]) / scaled_down_map.shape[1],
                )

                grid_x = int((agent_pos[1] - lower_bound[0]) / grid_size[0])
                grid_y = int((agent_pos[0] - lower_bound[2]) / grid_size[1])

                # set start and end pose for A*
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
                        0,
                        # self.current_observations[idx][0]["position"]["position"][1],
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
                # Retrieve observation
                obs = self.current_observations[idx]
                objects_annotations = self.envs.call_at(0, "get_semantic_annotations")  # TODO: ask why '0' and not idx
                env_scene = self.envs.call_at(0, "get_scene")  # TODO: ask why '0' and not idx
                obs['bbsgt'] = self.object_detector_gt.get_observation(self.current_observations[idx]['semantic'],
                                                                       objects_annotations, env_scene)
                # Extract bounding box from observation
                bboxes = obs['bbsgt']
                # if len(bboxes["instances"].infos) != 0:
                #     # Crop RGB based on bounding boxes
                #     crop_scene = obs["rgb"]
                #     for bbox in bboxes:
                #         ymin, xmin, ymax, xmax = bbox
                #         crop_scene = obs["rgb"][ymin:ymax, xmin:xmax, :]
                #     # Captioner inference
                #     caption = self.captioner(crop_scene)["text"]
                #     print(caption)
                #     obs["caption"] = caption
                cv2.imshow("RGB", cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)
                crop_scene = torch.tensor(np.expand_dims(obs["rgb"], axis=0))
                obs["caption"] = self.captioner(crop_scene)["text"]
                obs["goals"] = self.sub_goals[idx]
                episode = self.envs.current_episodes()[idx]
                # n_step = self.envs.call_at(idx, "get_step")
                n_step = self.current_steps  # TODO: ask why not self.envs.call_at(idx, "get_step")
                paths = save_obs(self.exp_path, episode.episode_id, obs, n_step)
                generated_observations_paths.append(paths)
            for idx in range(self.envs.num_envs):
                done = self.current_dones[idx]
                self.num_steps_done += 1
                if done:
                    self.current_steps[idx] = 0

            if self.num_steps_done % 10 == 0:
                print(f"Exploration at {int(self.percent_done() * 100)}%")
        del self.current_dones
        del self.current_observations

        gc.collect()
        self.envs.close()
        return sorted(generated_observations_paths)
