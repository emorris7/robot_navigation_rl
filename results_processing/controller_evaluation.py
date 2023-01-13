
from ray.rllib.algorithms.sac import SACConfig
from PathFinder import GoalCallbacksCO, set_seeds
from env.SimpleEnvironment_waypoints import SimpleRobotEnvironmentWP


horizon_val = 600 
SEED = 8192
model_class = SimpleRobotEnvironmentWP
algo = (
        SACConfig()
        .rollouts(num_rollout_workers=8,horizon=horizon_val)
        .resources(model_class, env_config={"horizon":horizon_val, "render_mode":"rgb_array"})
        .callbacks(GoalCallbacksCO)
        .framework(framework="torch")
        # Seed for reproducibility and statistical significance
        .debugging(seed=SEED)
        .build()
    )

algo.restore("/Users/emilymorris/ray_results/SAC_SimpleRobotEnvironmentWP_2023-01-11_16-00-50k1ew5bbn/checkpoint_002401/")

set_seeds(SEED)

distance_vals = [0.3, 0.7, 0.9]
num_obstacles = [1, 2, 4, 6]
num_evals = 5000.0

env = model_class()

for dist in distance_vals:
    env.init_distance_from_goal = dist
    for num_obs in num_obstacles:
        env.num_obstacles = num_obs

        success = 0
        crash = 0
        num_steps = 0

        for i in range(num_evals):
            obs = env.reset()
            done = False
            steps = 0
            info = {}
            while not done:
                steps += 1
                action = algo.compute_single_action(obs)
                obs, reward, done, info = env.step(action)
            success += info["Success"]
            crash += info["Crash"]
            num_steps += steps
            
        mean_success = success / num_evals
        mean_crash = crash / num_evals
        average_steps = num_steps /num_evals
        with open('evaluation_values.txt', 'a') as f:
            print("Writing to file")
            f.write(str(dist)+ " " + str(num_obs) + " " + str(mean_success) + " " + str(mean_crash) + " " + str(average_steps) + "\n")