# type: ignore
import math
import os
import time
from collections import deque
from typing import *

import gym
from habitat.tasks.nav.nav import NavigationGoal, NavigationTask
from habitat.utils.visualizations import maps
from habitat_baselines.agents.simple_agents import GoalFollower
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensorboard_utils import TensorboardWriter

from experimenting_env.agents.model import *
from experimenting_env.agents.ppo import *
from experimenting_env.detector.model import MultiStageModel
from experimenting_env.utils.sensors_utils import save_obs
from experimenting_env.utils.skeleton import *
from experimenting_env.utils.storage import *

from ..baselines import Baseline


@baseline_registry.register_trainer(name="goalexplorationbaseline-v3")
class GoalExplorationTrajectoryImagesBaseline(Baseline):
    def __init__(self, config, exp_base_dir, detectron_args, **kwargs):
        super().__init__(config, exp_base_dir, GoalFollower, **kwargs)
        self.config = config
        # reduced canonical map size for network input
        self.map_width, self.map_height = 128, 128
        self.visualize = self.config.ppo.visualize  # visualize maps

        self.object_detector = ObjectDetectorEnv(
            {"cfg": detectron_args}, device="cuda:0"
        )

        # cumulative rewards and inputs over agent trajectories
        self.trajectory_rewards = None
        self.trajectory_inputs = None
        self.num_trajectory_inputs = 4 # history size
        self.trajectory_inputs_idx = 0 # current pointer into history

        print("GOAL EXPLORATION IMAGES TRAJECTORY INPUTS")

    def _step(self, envs):
        super()._step(envs)
        self.predict_current_bbs_and_update_pcd()

    def predict_current_bbs_and_update_pcd(self, detector_batch_size=8):
        # predict bbs with object detector

        for idx in range(0, self.envs.num_envs, detector_batch_size):
            max_idx = min(self.envs.num_envs, idx + detector_batch_size)
            images = [
                self.current_observations[i][0]['rgb'] for i in range(idx, max_idx)
            ]
            preds = self.object_detector.predict_batch(images)

            for i in range(max_idx - idx):
                current_env = idx + i

                self.current_observations[current_env][0]['bbs'] = preds[0][
                    i
                ]  # dropping feature vectors

                self.envs.call_at(
                    current_env,
                    "_update_pointcloud",
                    {"observations": self.current_observations[current_env]},
                )
                disagreement_map = self.envs.call_at(
                    current_env, "get_and_update_disagreement_map"
                )
                self.current_observations[current_env][0][
                    'disagreement_map'
                ] = disagreement_map

    def add_rewards(self):
        new_rewards = torch.from_numpy(
            np.array(
                [[
                    self.envs.call_at(idx, "get_last_reward")
                    for idx in range(self.envs.num_envs)
                ]]
            )
        )
        if self.trajectory_rewards is None:
            self.trajectory_rewards = new_rewards.detach().clone()
            return self.trajectory_rewards
        else:
            return torch.cat((self.trajectory_rewards, new_rewards), 0)

    def average_rewards(self):
        if self.trajectory_rewards is None:
            return torch.from_numpy(
                np.array(
                    [
                        0.0
                        for idx in range(self.envs.num_envs)
                    ]
                )
            )
        return torch.mean(self.trajectory_rewards, 0)

    def get_rewards(self):
        return torch.from_numpy(
            np.array(
                [
                    self.envs.call_at(idx, "get_last_reward")
                    for idx in range(self.envs.num_envs)
                ]
            )
        )

    def create_policy_inputs(self):

        print("creating policy inputs...")

        # init trajectory_inputs
        if self.trajectory_inputs == None:
            self.trajectory_inputs = torch.zeros(self.envs.num_envs, self.num_trajectory_inputs * 2, self.map_width, self.map_height)

        # create disagreeemt inputs
        disagreement_inputs = [
            self.current_observations[idx][0]['disagreement_map']
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
                self.current_observations[idx][0]['position']['position'][2],
                self.current_observations[idx][0]['position']['position'][0],
            ]

            position_input = cv2.resize(
                self.current_infos[idx]["top_down_map"]["map"],
                (
                    self.current_observations[idx][0]['disagreement_map'].shape[1],
                    self.current_observations[idx][0]['disagreement_map'].shape[0],
                ),
            )

            lower_bound, upper_bound = self.envs.call_at(idx, "get_map_bounds")

            grid_size = (
                abs(upper_bound[2] - lower_bound[2]) / position_input.shape[0],
                abs(upper_bound[0] - lower_bound[0]) / position_input.shape[1],
            )

            grid_x = int((agent_pos[1] - lower_bound[0]) / grid_size[0])
            grid_y = int((agent_pos[0] - lower_bound[2]) / grid_size[1])

            # draw agent position as circle
            cv2.circle(position_input, (grid_x, grid_y), 20, 0, -1)
            cv2.circle(position_input, (grid_x, grid_y), 20, 2, 3)

            position_input = cv2.resize(
                position_input, (self.map_height, self.map_width)
            ).reshape(1, 1, self.map_height, self.map_width)
            position_inputs.append(position_input)

        position_inputs = torch.tensor(position_inputs).reshape(
            self.envs.num_envs, 1, self.map_height, self.map_width
        )

        # concat inputs
        inputs = torch.cat((disagreement_inputs, position_inputs), dim=1)


        # populate new inputs
        try:
            self.trajectory_inputs[:, self.trajectory_inputs_idx * 2: (self.trajectory_inputs_idx + 1) * 2, :, :] = inputs
        except:
            print("ERROR in populating inputs!!!")
            print("input shape:",inputs.shape, "global_input shape:", self.trajectory_inputs.shape,"idx:",self.trajectory_inputs_idx)

        # vis = np.zeros((self.map_height, self.map_width), dtype=np.uint8)
        # for i in range(self.num_trajectory_inputs * 2):
        #     img = (self.trajectory_inputs[0, i, :, :] * 255).cpu().numpy().astype(np.uint8)
        #     vis = np.concatenate((vis, img), axis=1)
        # cv2.imshow("inputs agent 0",vis[:,self.map_width:-1])
        # cv2.waitKey(10)    

        return self.trajectory_inputs

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

            # self.trajectory_rewards = self.add_rewards()

            # populate inputs
            # curr history idx, map, position input
            self.global_input = self.create_policy_inputs()

            self.trajectory_inputs_idx += 1
            if self.trajectory_inputs_idx == self.num_trajectory_inputs:
                frame = self.trajectory_inputs[:, 2:(self.num_trajectory_inputs)*2, :, :].clone().detach()
                self.trajectory_inputs[:, 0:(self.num_trajectory_inputs-1)*2, :, :] = frame
                self.trajectory_inputs_idx = self.num_trajectory_inputs - 1

    def predict_new_goals(self):
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
                        * self.current_observations[i][0]['disagreement_map'].shape[1]
                    ),
                    int(
                        action[1]
                        * self.current_observations[i][0]['disagreement_map'].shape[0]
                    ),
                ]
                for i, action in enumerate(cpu_actions)
            ]
            # compute subgoals for each goal
            for idx, rescaled_pixel_goal in enumerate(rescaled_pixel_goals):
                # convert pixels to global map coords
                lower_bound, upper_bound = self.envs.call_at(idx, "get_map_bounds")
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
                    self.current_observations[idx][0]['position']['position'][2],
                    self.current_observations[idx][0]['position']['position'][0],
                ]

                scaled_down_map = cv2.resize(
                    mymaps[idx],
                    (
                        self.current_observations[idx][0]['disagreement_map'].shape[1],
                        self.current_observations[idx][0]['disagreement_map'].shape[0],
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
                            self.current_observations[idx][0]['disagreement_map'].shape[
                                1
                            ],
                            self.current_observations[idx][0]['disagreement_map'].shape[
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
                        self.current_observations[idx][0]['position']['position'][1],
                        realworld_subgoal_y,
                    ]

                    self.sub_goals[idx].append(realworld_subgoal)

                self.sub_goals[idx].pop(-1)

                if len(self.sub_goals[idx]) > 0:
                    self.got_new_plan[idx] = True
                else:
                    print("A* failed, no waypoints")

            # clear previous policy inputs and rewards
            self.trajectory_inputs = torch.zeros(self.envs.num_envs, self.num_trajectory_inputs * 2, self.map_width, self.map_height)
            self.trajectory_rewards = None
            self.trajectory_inputs_idx = 0

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

        # goal policy observation space
        self.g_observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.num_trajectory_inputs * 2, self.map_width, self.map_height), dtype='uint8'
        )

        # goal policy action space
        self.g_action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # goal policy init
        device = torch.device("cuda:0") # if self.config.ppo.cuda else "cpu")

        self.g_policy = RL_Policy(
            self.g_observation_space.shape,
            self.g_action_space,
            1,
            base_kwargs={
                'recurrent': self.config.ppo.use_recurrent_global,
                'hidden_size': self.config.ppo.g_hidden_size,
                'downscaling': self.config.ppo.global_downscaling,
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
            self.envs.num_envs, self.num_trajectory_inputs * 2, self.map_height, self.map_width
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

            start = time.time()

            # step all envs
            self._step(self.envs)
            print('env step time:', time.time() - start)
            start = time.time()
           
            not_ready = False
            # if no observations or no disagr map, exit
            if not self.current_infos or not self.current_observations:
                print("\n\n\nINFOS OR OBS NOT READY")
                not_ready = True

            for idx in range(self.envs.num_envs):
                if 'disagreement_map' not in self.current_observations[idx][0]:
                    print("\n\n\nDISAGREEMENT NOT READY FOR IDX:", idx)
                    not_ready = True

            if not_ready:
                continue


            # for each env, visualize and send next subgoal if needed
            for idx in range(self.envs.num_envs):
                obs = self.current_observations[idx]
                info = self.current_infos[idx]
                episode = self.envs.current_episodes()[idx]
                paths = save_obs(
                    self.exp_path, episode.episode_id, obs, self.current_steps[idx]
                )

                # visualize maps for debugging purposes
                if self.visualize and self.astar_img[idx] is not None:
                    cv2.imshow("Env " + str(idx) + " A*", self.astar_img[idx])

                if self.visualize and ('disagreement_map' in obs[0]):
                    rgb_map = maps.colorize_draw_agent_and_fit_to_height(
                        info["top_down_map"], obs[0]['disagreement_map'].shape[0]
                    )
                    cv2.imshow("Env " + str(idx) + " map", rgb_map)
                    cv2.imshow(
                        "Env " + str(idx) + " disagreement", obs[0]['disagreement_map']
                    )

                # if time to go to next subgoal, do it
                self.goto_next_subgoal(idx)

            print('next subgoal time:', time.time() - start)
            start = time.time()

            if self.visualize:
                cv2.waitKey(10)

            # get reward
            self.g_reward = self.get_rewards()

            print('get rewards time:', time.time() - start)
            start = time.time()

            # orientation input
            agent_orient = [
                self.current_observations[idx][0]['position']['orientation']
                for idx in range(self.envs.num_envs)
            ]
            agent_orientation = [2.0 * math.acos(q.y) for q in agent_orient]
            self.global_orientation = torch.tensor(agent_orientation).reshape(
                self.envs.num_envs, 1
            )

            # if time to predict new goal
            self.predict_new_goals()

            print('prediction time:', time.time() - start)
            start = time.time()

            for idx in range(self.envs.num_envs):
                done = self.current_dones[idx]


                generated_observations_paths.append(paths)
                self.num_steps_done += 1
                if done:
                    self.current_steps[idx] = 0

            print('gen data time:', time.time() - start)
            start = time.time()

            if self.num_steps_done % 10 == 0:
                print(f"Exploration at {int(self.percent_done() * 100)}%")
        del self.current_dones
        del self.current_observations

        self.envs.close()
        return sorted(generated_observations_paths)

    def _training_log(
            self,
            writer,
            reward: float,
            losses: Dict[str, float],
            metrics: Dict[str, float] = None,
            prev_time: int = 0,
    ):
        writer.add_scalar(
            "reward",
            reward,
            self.num_steps_done,
        )

        writer.add_scalars(
            "losses",
            losses,
            self.num_steps_done,
        )

        if metrics:
            writer.add_scalars("metrics", metrics, self.num_steps_done)

    def train(self) -> None:
        self.envs = self._init_train()
        self.sub_goals = [[] for _ in range(self.envs.num_envs)]
        self.sub_goals_counter = [0] * self.envs.num_envs
        self.got_new_plan = [False] * self.envs.num_envs
        self.replan_retries = [0] * self.envs.num_envs
        self.replan = [True] * self.envs.num_envs
        self.astar_img = [None] * self.envs.num_envs

        self.first_step = True

        # goal policy observation space
        self.g_observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.num_trajectory_inputs * 2, self.map_width, self.map_height), dtype='uint8'
        )

        # goal policy action space
        self.g_action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # goal policy init
        device = torch.device("cuda:0") # if self.config.ppo.cuda else "cpu")

        self.g_policy = RL_Policy(
            self.g_observation_space.shape,
            self.g_action_space,
            1,
            base_kwargs={
                'recurrent': self.config.ppo.use_recurrent_global,
                'hidden_size': self.config.ppo.g_hidden_size,
                'downscaling': self.config.ppo.global_downscaling,
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
            self.envs.num_envs, self.num_trajectory_inputs * 2, self.map_height, self.map_width
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

        self.t_start = time.time()  # for tensorboard
        writer = TensorboardWriter(self.config.TENSORBOARD_DIR, flush_secs=30)

        self.checkpoint_folder = self.config.CHECKPOINT_FOLDER

        if not os.path.exists(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)

        # self.g_policy.load_state_dict(torch.load("/home/morpheus/code/cvpr2022/batchedpolicy/unsupervised-sensor-network/exps/pipeline_soft/02-24-11-58-20_habitat_gibson_goal_exploration.yaml_pipeline_soft_consensus_logits_batchsize_16_0.001_contrastive_0.0/checkpoints/checkpoint_797.ckpt"))
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
                if 'disagreement_map' not in self.current_observations[idx][0]:
                    print("\n\n\nDISAGREEMENT NOT READY FOR IDX:", idx)
                    not_ready = True

            if not_ready:
                continue

            # for each env, visualize and send next subgoal if needed
            for idx in range(self.envs.num_envs):
                obs = self.current_observations[idx]
                info = self.current_infos[idx]
                episode = self.envs.current_episodes()[idx]
                # print(f"Episode {episode.episode_id} for env")

                # visualize maps for debugging purposes
                if self.visualize and self.astar_img[idx] is not None:
                    cv2.imshow("Env " + str(idx) + " A*", self.astar_img[idx])

                if self.visualize and ('disagreement_map' in obs[0]):
                    rgb_map = maps.colorize_draw_agent_and_fit_to_height(
                        info["top_down_map"], obs[0]['disagreement_map'].shape[0]
                    )
                    cv2.imshow("Env " + str(idx) + " map", rgb_map)
                    cv2.imshow(
                        "Env " + str(idx) + " disagreement", obs[0]['disagreement_map']
                    )

                # is it time to go to next subgoal?
                self.goto_next_subgoal(idx)

            if self.visualize:
                cv2.waitKey(10)

            # get reward
            self.g_reward = self.get_rewards()

            # last input was generated when selecting next subgoal
            # self.global_input = self.trajectory_inputs

            # orientation input
            agent_orient = [
                self.current_observations[idx][0]['position']['orientation']
                for idx in range(self.envs.num_envs)
            ]
            agent_orientation = [2.0 * math.acos(q.y) for q in agent_orient]
            self.global_orientation = torch.tensor(agent_orientation).reshape(
                self.envs.num_envs, 1
            )

            self.predict_new_goals()

            # train goal policy
            if self.num_steps_done % self.config.ppo.num_global_steps == 0:
                print("ppo training step")

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
                reward = self.envs.call_at(idx, "get_last_reward")
                self._training_log(writer, reward, losses)

                # self.g_reward = self.average_rewards()

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


class ObjectDetectorEnv:
    def __init__(self, config, device="cpu"):

        self.object_detector = MultiStageModel(**config)
        self.device = device

        self.object_detector.to(device)
        self.object_detector.eval()
        self.object_detector.model.roi_heads.box_predictor.box_predictor.test_score_thresh = (
            0.5
        )

    def load(self, load_path):
        if os.path.exists(load_path):
            self.object_detector.load_state_dict(torch.load(load_path)).to(self.device)

    def move_to_device(self, device='cpu'):
        self.device = device
        self.object_detector.to(device)

    def predict_batch(self, images):

        predictions = self.object_detector.infer(images)
        for i in range(len(images)):
            predictions[0][i]['instances'] = predictions[0][i]['instances'].to("cpu")

        return predictions
