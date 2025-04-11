import habitat
import gym 
import time
from habitat.gym.gym_wrapper import HabGymWrapper
from habitat.core.environments import RLTaskEnv
from habitat import RLEnv
from habitat import Dataset
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Type, Union
import numpy as np
from habitat.utils.visualizations import maps
from habitat_sim.utils import common as sim_utils
import magnum as mn
import math
from habitat.tasks.rearrange.rearrange_task import  RearrangeTask
from experimenting_env.utils import projection_utils as pu
from experimenting_env.utils.matching import get_objects_ids
import habitat_sim

@habitat.registry.register_env(name="Habitat3Env")
class EnvHabitat3(gym.Wrapper):
    """
    A registered environment that wraps a RLTaskEnv with the HabGymWrapper
    to use the default gym API.
    """

    def __init__(
        self, config: "DictConfig", dataset: Optional[Dataset] = None
    ):
        base_env = RLTaskEnv(config=config, dataset=dataset)
        # base_env = RearrangeTask(config=config, dataset=dataset)
        env = HabGymWrapper(env=base_env)
        super().__init__(env)
        self.pcd = None
        self.disagreement_map = None
        self.last_reward = 0.0
        self._episode_sum_reward = 0.0
        self._max_episode_steps = config.environment.max_episode_steps

    def reset(self):
        self._elapsed_steps = 0
        self.episode_over = False

        return super().reset()
        
    def get_semantic_annotations(self):
        
        objects_annotation = self.env._sim.semantic_annotations().objects

        mapping = {
            int(obj.id.split("_")[-1]): obj.category.name()
            for obj in objects_annotation
            if obj is not None
        }
        return mapping

    def get_scene(self):
        return self.env._sim.habitat_config.scene

    def get_episode_id(self):
        return self.env.current_episode().episode_id

    def get_agent_position(self):
        return {
            'position': self.env._sim.get_agent_state().position,
            'orientation': self.env._sim.get_agent_state().rotation,
        }

    def get_upper_and_lower_map_bounds(self):
       lower_bounds, upper_bounds = self.env._sim.pathfinder.get_bounds()
       return [lower_bounds, upper_bounds]

    def set_goals(self, data):
        self.current_goal = data

    def get_reward(self, disagreement_map):
        disagreement_reward = 0

        if disagreement_map is not None:
            disagreement_reward = disagreement_map.sum() / 1000.0

        self.last_reward = disagreement_reward
        self._episode_sum_reward += disagreement_reward

        return disagreement_reward

    def _past_limit(self) -> bool:
        if (
            self._max_episode_steps != 0
            and self._max_episode_steps <= self._elapsed_steps
        ):
            return True

        return False
        
    def get_done(self, observations):
        self.episode_over = False

        info = self.habitat_env.get_metrics()
        allo_map = info['top_down_map']['map']
        allo_map[allo_map > 1] = 1
        expl_map = info['top_down_map']['fog_of_war_mask']

        unknown_map = allo_map - expl_map
        freespace_area = np.sum(allo_map[allo_map == 1])
        unknown_area = np.sum(unknown_map[unknown_map > 0])
        unknown_percent = unknown_area / freespace_area

        # if unknown_percent < 0.1:
        #     print("Exploration task completed")
        #     self.episode_over = True

        self.episode_over = self._past_limit()
        
        return self.episode_over

    def _episode_success(self, observations):
        r"""Returns True if within distance threshold of the goal."""
        return False
        
    def get_info(self, observations):
        info = self.habitat_env.get_metrics()
        info["episode_success"] = self._episode_success(observations)
        info["episode_reward"] = self._episode_sum_reward / self._max_episode_steps

        expl_map = info['top_down_map']['fog_of_war_mask']
        allo_map = info['top_down_map']['map']
        allo_map[allo_map > 1] = 1
        info["area_ratio"] = expl_map.sum() / (allo_map > 0).sum()

        self.area_ratio = expl_map.sum() / (allo_map > 0).sum()

        return info

#     def step(self, action):
#         observations = [super().step(action)]
# 
#         done = self.get_done(observations)
#         info = self.get_info(observations)
#         
#         # print("-----------------------------", observations[0][0].keys(), observations[0][1], type(observations[0][2]))
#         
#         if "disagreement_map" in observations[0]:
#             reward = (
#                 self.get_reward(observations[0]["disagreement_map"])
#             )  # get reward uses current pcd for getting reward; if the pcd is update (e.g., with _update_pointcloud, after getting current observations) you need to call get_reward again
#         else:
#             reward = 0.0
#             
#         return observations, reward, done, info
           
    def update_pointcloud(self, observations, episode_id, map_bounds):
        instances = observations["bbs"]
        preds = instances["instances"].to("cpu")
        batch = observations["bbs"]
        batch["episode"] = int(episode_id)

        infos = get_objects_ids([batch], [preds])[0]

        preds.infos = infos

        pose = observations["position"]

        point_cloud = pu.project_semantic_masks_to_3d(
            observations["depth"].squeeze(),
            pose,
            preds,
            infos,
        )
        point_cloud._episode = int(episode_id)

        lower_bound, upper_bound = map_bounds
        if self.pcd is None or self.pcd._episode != point_cloud._episode:
            self.pcd: pu.SemanticPointCloud = point_cloud

        else:
            point_cloud.preprocess(lower_bound, upper_bound)

            self.pcd += point_cloud

        if len(preds) == 0:
            return

        t = time.time()
        self.pcd.preprocess(lower_bound, upper_bound)
        
    def get_and_update_disagreement_map(self, map_bounds):
        lower_bound, upper_bound = map_bounds

        disagreement_map = self.pcd.get_topdown_semantic(lower_bound, upper_bound)[
            :, :, -1
        ]
        self.disagreement_map = disagreement_map
        return disagreement_map
    
    def get_step(self):
        return self.env.elapsed_steps
    
    def get_path(self, agent_position, goal):
        pathfinder = self.habitat_env.sim.pathfinder
        if not pathfinder.is_loaded:
            print("Pathfinder not initialized, aborting.")
        else:
            path = habitat_sim.ShortestPath()
            path.requested_start = agent_position.reshape(3, 1)
            path.requested_end = pathfinder.snap_point(goal.reshape(3, 1))
            found_path = pathfinder.find_path(path)
            # print("Found path:", found_path)
        return np.array(path.points)

