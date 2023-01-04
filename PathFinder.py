from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from env.SimpleEnvironment import SimpleRobotEnviroment
from env.SimpleEnvironment import GOAL_DISTANCE, GOAL_ANGLE
import ray.rllib.utils as u
from gym.wrappers import TimeLimit
from gym import make
from ray.tune.registry import register_env
from gym.envs.registration import register
import numpy as np

# for the custom callback
from typing import Dict
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.algorithms.callbacks import DefaultCallbacks


# env = SimpleRobotEnviroment(render_mode="rgb_array") 
# print(env.spec)

# # u.check_env(SimpleRobotEnviroment(render_mode="rgb_array"))

# def env_creator(env_config):
#     register(
#         id='simple-robot-env-v01',
#         entry_point='SimplePathFinder.env:SimpleRobotEnviroment',
#         max_episode_steps=200,
#     )
#     env = make("simple-robot-env-v01")
#     # env = SimpleRobotEnviroment(render_mode=env_config["render_mode"])
#     # env = TimeLimit(env, max_episode_steps=100)
#     return env  # return an env instance

# register_env("simple-robot-env-v01", env_creator)
# env = make('simple-robot-env-v0')

class GoalCallbacks(DefaultCallbacks):

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        final_x = episode.last_observation_for()[0]
        final_y = episode.last_observation_for()[1]
        final_yaw = episode.last_observation_for()[2]
        goal_x = episode.last_observation_for()[3]
        goal_y = episode.last_observation_for()[4]
        goal_yaw = episode.last_observation_for()[5]
        # reached_goal = (final_distance <= GOAL_DISTANCE) and (final_angle <= GOAL_ANGLE)
        # print("episode {} ended with length {} and reached goal is {}".format(
        #     episode.episode_id, episode.length, reached_goal))
        episode.custom_metrics["final_distance"] = np.linalg.norm(np.array([goal_x, goal_y]) - np.array([final_x,final_y]))
        episode.custom_metrics["final_angle_difference"] = min(np.abs(goal_yaw - final_yaw), 2*np.pi - np.abs(goal_yaw - final_yaw))


if __name__ == '__main__':

    algo = (
        PPOConfig()
        .rollouts(num_rollout_workers=1,horizon=200)
        .resources(num_gpus=0)
        .environment(SimpleRobotEnviroment, env_config={"render_mode":"rgb_array"})
        .callbacks(GoalCallbacks)
        .build()
    )

    # print(algo.config.horizon)
    # print("m")

    for i in range(200):
        print(i)
        result = algo.train()
        # print(result["custom_metrics"])
        # print(pretty_print(result))

        if i % 5 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")