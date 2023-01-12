from typing import Optional
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from env.SimpleEnvironment import SimpleRobotEnviroment
from ray.rllib.env.env_context import EnvContext
from ray.rllib.algorithms.sac import SACConfig
from PathFinder import GoalCallbacks, set_seeds
import numpy as np

TaskType = dict  # Can be different types depending on env, e.g., int or dict
SEED = 4096

class CurriculumSimpleEnvironment(SimpleRobotEnviroment, TaskSettableEnv):

    def __init__(self, config: Optional[EnvContext] = None):
        super().__init__(config)

    def set_task(self, task: TaskType) -> None:
        print(task)
        self.goal_tolerance = task["goal_tolerance"]
        self.tolerance_step_change = task["tolerance_step_change"]

    def get_task(self) -> TaskType:
        task_dict = {"goal_tolerance":self.goal_tolerance, "tolerance_step_change":self.tolerance_step_change}
        return task_dict

def curriculum_fn(train_results, task_settable_env, env_ctx):
    success = train_results["custom_metrics"]["reached_goal_mean"]
    timesteps = train_results["timesteps_total"]
    timestep_diff = np.abs(task_settable_env.tolerance_step_change - timesteps)
    # print(timesteps)
    if success >= 0.92 and timestep_diff >= 20000:
        print("CHANGING LEARNING TOLERANCE LEVEL")
        new_tolerance = task_settable_env.goal_tolerance / 2
        task_dict = {"goal_tolerance":new_tolerance, "tolerance_step_change":timesteps}
        return task_dict
    else:
        return task_settable_env.get_task()

if __name__ == '__main__':
    horizon_val = 300 
    algo = (
        SACConfig()
        .rollouts(num_rollout_workers=8,horizon=horizon_val)
        .resources(num_gpus=0)
        .environment(CurriculumSimpleEnvironment, env_task_fn=curriculum_fn, env_config={"horizon":horizon_val, "render_mode":"rgb_array"})
        .callbacks(GoalCallbacks)
        .framework(framework="torch")
        # Seed for reproducibility and statistical significance
        .debugging(seed=SEED)
        .build()
    )

    set_seeds(SEED)

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
