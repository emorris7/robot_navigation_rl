from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.mbmpo import MBMPOConfig
from ray.tune.logger import pretty_print
from env.SimpleEnvironment import SimpleRobotEnviroment
from env.SimpleEnvironment_condensed_obs import SimpleRobotEnviromentCO
from env.SimpleEnvironment_waypoints import SimpleRobotEnvironmentWP
import numpy as np
import torch
import random
from ray.rllib.algorithms.algorithm import Algorithm

# for the custom callback
from typing import Dict
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.algorithms.callbacks import DefaultCallbacks

SEED = 4096

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
        success = episode.last_info_for()["Success"]
        crash = episode.last_info_for()["Crash"]

        episode.custom_metrics["final_distance"] = np.linalg.norm(np.array([goal_x, goal_y]) - np.array([final_x,final_y]))
        episode.custom_metrics["final_angle_difference"] = min(np.abs(goal_yaw - final_yaw), 2*np.pi - np.abs(goal_yaw - final_yaw))
        episode.custom_metrics["reached_goal"] = success
        episode.custom_metrics["crash"] = crash

        
class GoalCallbacksCO(DefaultCallbacks):

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        final_distance = episode.last_observation_for()[0]
        final_angle_diff = abs(episode.last_observation_for()[2])
        success = episode.last_info_for()["Success"]
        crash = episode.last_info_for()["Crash"]
        # goal_yaw = episode.last_observation_for()[3]
        # final_yaw = episode.last_observation_for()[2]
        # final_angle_diff = min(np.abs(goal_yaw - final_yaw), 2*np.pi - np.abs(goal_yaw - final_yaw))

        episode.custom_metrics["final_distance"] = final_distance
        episode.custom_metrics["final_angle_difference"] = final_angle_diff
        episode.custom_metrics["reached_goal"] = success
        episode.custom_metrics["crash"] = crash

def set_seeds(seed):
    torch.manual_seed(seed)  # Sets seed for PyTorch RNG
    torch.cuda.manual_seed_all(seed)  # Sets seeds of GPU RNG
    np.random.seed(seed=seed)  # Set seed for NumPy RNG
    random.seed(seed)

if __name__ == '__main__':

    # algo = (
    #     PPOConfig()
    #     # .training(lr=1e-4)
    #     # .training(model={'use_lstm':True})
    #     # .training(train_batch_size=60000, sgd_minibatch_size=4096)
    #     # Increase horizon from 200 to 400 as robot was ending before reaching goal
    #     .rollouts(num_rollout_workers=1,horizon=600)
    #     .resources(num_gpus=0)
    #     .environment(SimpleRobotEnviroment, env_config={"render_mode":"rgb_array"})
    #     .callbacks(GoalCallbacks)
             # Seed for reproducibility and statistical significance
    #     .debugging(seed=SEED)
    #     .build()
    # )

    # algo = (
    #     PPOConfig()
    #     # .training(lr=1e-4)
    #     # .training(model={'use_lstm':True})
    #     # .training(train_batch_size=60000, sgd_minibatch_size=4096)
    #     # Increase horizon from 200 to 400 as robot was ending before reaching goal
    #     .rollouts(num_rollout_workers=1,horizon=600)
    #     .resources(num_gpus=0)
    #     .environment(SimpleRobotEnviromentCO, env_config={"render_mode":"rgb_array"})
    #     .callbacks(GoalCallbacksCO)
             # Seed for reproducibility and statistical significance
    #     .debugging(seed=SEED)
    #     .build()
    # )
    # print(algo.config.horizon)
    # print("m")
    

    horizon_val = 600 
    # algo = (
    #     SACConfig()
    #     .rollouts(num_rollout_workers=8,horizon=horizon_val)
    #     .resources(num_gpus=0)
    #     .environment(SimpleRobotEnviromentCO, env_config={"horizon":horizon_val,"render_mode":"rgb_array"})
    #     .callbacks(GoalCallbacksCO)
    #     .framework(framework="torch")
    #     # Seed for reproducibility and statistical significance
    #     .debugging(seed=SEED)
    #     .build()
    # )

    
    # algo = (
    #     SACConfig()
    #     .rollouts(num_rollout_workers=8,horizon=horizon_val)
    #     .resources(num_gpus=0)
    #     .environment(SimpleRobotEnviroment, env_config={"horizon":horizon_val, "render_mode":"rgb_array"})
    #     .callbacks(GoalCallbacks)
    #     .framework(framework="torch")
    #     # Seed for reproducibility and statistical significance
    #     .debugging(seed=SEED)
    #     .build()
    # )

    algo = (
        SACConfig()
        .rollouts(num_rollout_workers=8,horizon=horizon_val)
        .resources(num_gpus=0)
        .environment(SimpleRobotEnvironmentWP, env_config={"horizon":horizon_val, "render_mode":"rgb_array"})
        .callbacks(GoalCallbacksCO)
        .framework(framework="torch")
        # Seed for reproducibility and statistical significance
        .debugging(seed=SEED)
        .build()
    )
    
    # # For testing
    # algo.restore("/Users/emilymorris/ray_results/SAC_SimpleRobotEnviromentCO_2023-01-11_10-25-31cu3swts5/checkpoint_003411/")

    # num_episodes = 6000
    # for i in range(num_episodes):
    #     print(i)
    #     result = algo.train()
    #     # print(result["custom_metrics"])
    #     # print(pretty_print(result))

    #     if i % 10 == 0 or i==num_episodes-1:
    #         checkpoint_dir = algo.save()
    #         print(f"Checkpoint saved in directory {checkpoint_dir}")

    # Set all our seeds for the environment
    set_seeds(seed=SEED)

    i = 0
    while True:
        print(i)
        result = algo.train()
        # print(result["custom_metrics"])
        # print(pretty_print(result))

        if i % 10 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")
        
        i+=1

    
    # env = SimpleRobotEnviromentCO()
    # obs = env.reset()
    # done = False
    # print(obs)

    # import matplotlib.pyplot as plt
    # def displayImage(image):
    #     plt.imshow(image)
    #     plt.axis('off')
    #     plt.show()

    # x = env.render()
    # displayImage(x)
    
    # for i in range(300):
    #     print(i)
    #     if not done:
    #         action = algo.compute_single_action(obs)
    #         print("Action: ", action)
    #         obs, reward, done, _ = env.step(action)
    #         print("ROBOT: ", env.robot.pose)
    #         print("Observation:",obs)
    #         print("Reward: ", reward)
    #         print("Done",done)
    #     else:
    #         print("Done")


    # x = env.render()
    # displayImage(x)