
from ray.rllib.algorithms.sac import SACConfig
from PathFinder import GoalCallbacksCO, set_seeds, GoalCallbacks
from env.SimpleEnvironment_condensed_obs import SimpleRobotEnviromentCO
from env.SimpleEnvironment import SimpleRobotEnviroment
import numpy as np

def run_evaluation():

    waypoint_1024_checkpoint = "/Users/emilymorris/ray_results/SAC_SimpleRobotEnvironmentWP_2023-01-13_17-03-57wu5205lh/checkpoint_001971"
    waypoint_2048_checkpoint = "/Users/emilymorris/ray_results/SAC_SimpleRobotEnvironmentWP_2023-01-11_15-17-1741kday90/checkpoint_002011"
    waypoint_4096_checkpoint = "/Users/emilymorris/ray_results/SAC_SimpleRobotEnvironmentWP_2023-01-11_16-00-50k1ew5bbn/checkpoint_001801"

    checkpoint_4096 = "/Users/emilymorris/ray_results/SAC_SimpleRobotEnviromentCO_2023-01-11_11-48-18_3vmn6e3/checkpoint_003501"
    checkpoint_2048 = "/Users/emilymorris/ray_results/SAC_SimpleRobotEnviromentCO_2023-01-10_23-12-41qecbjmyh/checkpoint_003161"
    checkpoint_1024 = "/Users/emilymorris/ray_results/SAC_SimpleRobotEnviromentCO_2023-01-11_09-15-29l59naez1/checkpoint_003501"
    
    # results with 0.03 tolerance
    curriculum = "/Users/emilymorris/ray_results/SAC_CurriculumSimpleEnvironment_2023-01-13_10-09-40czkn_fcg/checkpoint_021202"
    horizon_val = 600 
    SEED = 8192

    # algo = (
    #         SACConfig()
    #         .rollouts(num_rollout_workers=8,horizon=horizon_val)
    #         .resources(num_gpus=0)
    #         .environment(SimpleRobotEnviromentCO, env_config={"horizon":horizon_val, "render_mode":"rgb_array"})
    #         .callbacks(GoalCallbacksCO)
    #         .framework(framework="torch")
    #         # Seed for reproducibility and statistical significance
    #         .debugging(seed=SEED)
    #         .build()
    #     )

    algo = (
            SACConfig()
            .rollouts(num_rollout_workers=8,horizon=horizon_val)
            .resources(num_gpus=0)
            .environment(SimpleRobotEnviroment, env_config={"horizon":horizon_val, "render_mode":"rgb_array"})
            .callbacks(GoalCallbacks)
            .framework(framework="torch")
            # Seed for reproducibility and statistical significance
            .debugging(seed=SEED)
            .build()
        )

    algo.restore(curriculum)

    set_seeds(SEED)

    distance_vals = [0.3, 0.7, 0.9]
    num_obstacles = [1, 2, 4, 6]
    num_evals = 1000.0


    for dist in distance_vals:
        for num_obs in num_obstacles:
            # env = SimpleRobotEnviromentCO(num_obstacles=num_obs, init_distance=dist)
            env = SimpleRobotEnviroment(num_obstacles=num_obs, init_distance=dist)
            # Set goal tolerance to 0.03 for evaluating curriculum environment
            env.goal_tolerance = 0.03
            print(env.init_distance_from_goal)
            print(env.num_obstacles)

            success = 0
            crash = 0
            num_steps = 0
            final_dist = 0

            for i in range(int(num_evals)):
                print(i)
                obs = env.reset()
                done = False
                steps = 0
                info = {}
                while not done and steps <= horizon_val:
                    steps += 1
                    action = algo.compute_single_action(obs)
                    obs, reward, done, info = env.step(action)
                final_dist += obs[0]
                success += info["Success"]
                crash += info["Crash"]
                num_steps += steps
            
            mean_final_dist = final_dist / num_evals
            mean_success = success / num_evals
            mean_crash = crash / num_evals
            average_steps = num_steps /num_evals
            with open('evaluation_values.txt', 'a') as f:
                print("Writing to file")
                f.write(str(dist)+ " " + str(num_obs) + " " + str(mean_success) + " " + str(mean_crash) + " " + str(average_steps) + " " + str(mean_final_dist) + "\n")

def process_evalution(filenames):
    np_03_1_dist, np_03_1_success, np_03_1_crash, np_03_1_episode = [], [], [], []
    np_03_2_dist, np_03_2_success, np_03_2_crash, np_03_2_episode = [], [], [], []
    np_03_4_dist, np_03_4_success, np_03_4_crash, np_03_4_episode = [], [], [], []
    np_03_6_dist, np_03_6_success, np_03_6_crash, np_03_6_episode = [], [], [], []

    np_07_1_dist, np_07_1_success, np_07_1_crash, np_07_1_episode = [], [], [], []
    np_07_2_dist, np_07_2_success, np_07_2_crash, np_07_2_episode = [], [], [], []
    np_07_4_dist, np_07_4_success, np_07_4_crash, np_07_4_episode = [], [], [], []
    np_07_6_dist, np_07_6_success, np_07_6_crash, np_07_6_episode = [], [], [], []

    np_09_1_dist, np_09_1_success, np_09_1_crash, np_09_1_episode = [], [], [], []
    np_09_2_dist, np_09_2_success, np_09_2_crash, np_09_2_episode = [], [], [], []
    np_09_4_dist, np_09_4_success, np_09_4_crash, np_09_4_episode = [], [], [], []
    np_09_6_dist, np_09_6_success, np_09_6_crash, np_09_6_episode = [], [], [], []

    for f in filenames:
        with open(f, 'r') as file:
            print("Processing file")
            line_1 = file.readline().strip().split()
            print(line_1)
            np_03_1_dist.append(float(line_1[5])), np_03_1_success.append(float(line_1[2])), np_03_1_crash.append(float(line_1[3])), np_03_1_episode.append(float(line_1[4]))

            line_2 = file.readline().strip().split()
            np_03_2_dist.append(float(line_2[5])), np_03_2_success.append(float(line_2[2])), np_03_2_crash.append(float(line_2[3])), np_03_2_episode.append(float(line_2[4]))

            line_3 = file.readline().strip().split()
            np_03_4_dist.append(float(line_3[5])), np_03_4_success.append(float(line_3[2])), np_03_4_crash.append(float(line_3[3])), np_03_4_episode.append(float(line_3[4]))

            line_4 = file.readline().strip().split()
            np_03_6_dist.append(float(line_4[5])), np_03_6_success.append(float(line_4[2])), np_03_6_crash.append(float(line_4[3])), np_03_6_episode.append(float(line_4[4]))

            line_5 = file.readline().strip().split()
            np_07_1_dist.append(float(line_5[5])), np_07_1_success.append(float(line_5[2])), np_07_1_crash.append(float(line_5[3])), np_07_1_episode.append(float(line_5[4]))

            line_6 = file.readline().strip().split()
            np_07_2_dist.append(float(line_6[5])), np_07_2_success.append(float(line_6[2])), np_07_2_crash.append(float(line_6[3])), np_07_2_episode.append(float(line_6[4]))

            line_7 = file.readline().strip().split()
            np_07_4_dist.append(float(line_7[5])), np_07_4_success.append(float(line_7[2])), np_07_4_crash.append(float(line_7[3])), np_07_4_episode.append(float(line_7[4]))

            line_8 = file.readline().strip().split()
            np_07_6_dist.append(float(line_8[5])), np_07_6_success.append(float(line_8[2])), np_07_6_crash.append(float(line_8[3])), np_07_6_episode.append(float(line_8[4]))

            line_9 = file.readline().strip().split()
            np_09_1_dist.append(float(line_9[5])), np_09_1_success.append(float(line_9[2])), np_09_1_crash.append(float(line_9[3])), np_09_1_episode.append(float(line_9[4]))

            line_10 = file.readline().strip().split()
            np_09_2_dist.append(float(line_10[5])), np_09_2_success.append(float(line_10[2])), np_09_2_crash.append(float(line_10[3])), np_09_2_episode.append(float(line_10[4]))

            line_11 = file.readline().strip().split()
            np_09_4_dist.append(float(line_11[5])), np_09_4_success.append(float(line_11[2])), np_09_4_crash.append(float(line_11[3])), np_09_4_episode.append(float(line_11[4]))

            line_12 = file.readline().strip().split()
            np_09_6_dist.append(float(line_12[5])), np_09_6_success.append(float(line_12[2])), np_09_6_crash.append(float(line_12[3])), np_09_6_episode.append(float(line_12[4]))

    mean_success_03 = [np.mean(np_03_1_success), np.mean(np_03_2_success), np.mean(np_03_4_success), np.mean(np_03_6_success)]
    mean_crash_03 = [np.mean(np_03_1_crash), np.mean(np_03_2_crash), np.mean(np_03_4_crash), np.mean(np_03_6_crash)]
    mean_dist_03 = [np.mean(np_03_1_dist), np.mean(np_03_2_dist), np.mean(np_03_4_dist), np.mean(np_03_6_dist)]
    mean_episode_03 = [np.mean(np_03_1_episode), np.mean(np_03_2_episode), np.mean(np_03_4_episode), np.mean(np_03_6_episode)]

    mean_success_07 = [np.mean(np_07_1_success), np.mean(np_07_2_success), np.mean(np_07_4_success), np.mean(np_07_6_success)]
    mean_crash_07 = [np.mean(np_07_1_crash), np.mean(np_07_2_crash), np.mean(np_07_4_crash), np.mean(np_07_6_crash)]
    mean_dist_07 = [np.mean(np_07_1_dist), np.mean(np_07_2_dist), np.mean(np_07_4_dist), np.mean(np_07_6_dist)]
    mean_episode_07 = [np.mean(np_07_1_episode), np.mean(np_07_2_episode), np.mean(np_07_4_episode), np.mean(np_07_6_episode)]

    mean_success_09 = [np.mean(np_09_1_success), np.mean(np_09_2_success), np.mean(np_09_4_success), np.mean(np_09_6_success)]
    mean_crash_09 = [np.mean(np_09_1_crash), np.mean(np_09_2_crash), np.mean(np_09_4_crash), np.mean(np_09_6_crash)]
    mean_dist_09 = [np.mean(np_09_1_dist), np.mean(np_09_2_dist), np.mean(np_09_4_dist), np.mean(np_09_6_dist)]
    mean_episode_09 = [np.mean(np_09_1_episode), np.mean(np_09_2_episode), np.mean(np_09_4_episode), np.mean(np_09_6_episode)]

    std_success_03 = [np.std(np_03_1_success), np.std(np_03_2_success), np.std(np_03_4_success), np.std(np_03_6_success)]
    std_crash_03 = [np.std(np_03_1_crash), np.std(np_03_2_crash), np.std(np_03_4_crash), np.std(np_03_6_crash)]
    std_dist_03 = [np.std(np_03_1_dist), np.std(np_03_2_dist), np.std(np_03_4_dist), np.std(np_03_6_dist)]
    std_episode_03 = [np.std(np_03_1_episode), np.std(np_03_2_episode), np.std(np_03_4_episode), np.std(np_03_6_episode)]

    std_success_07 = [np.std(np_07_1_success), np.std(np_07_2_success), np.std(np_07_4_success), np.std(np_07_6_success)]
    std_crash_07 = [np.std(np_07_1_crash), np.std(np_07_2_crash), np.std(np_07_4_crash), np.std(np_07_6_crash)]
    std_dist_07 = [np.std(np_07_1_dist), np.std(np_07_2_dist), np.std(np_07_4_dist), np.std(np_07_6_dist)]
    std_episode_07 = [np.std(np_07_1_episode), np.std(np_07_2_episode), np.std(np_07_4_episode), np.std(np_07_6_episode)]

    std_success_09 = [np.std(np_09_1_success), np.std(np_09_2_success), np.std(np_09_4_success), np.std(np_09_6_success)]
    std_crash_09 = [np.std(np_09_1_crash), np.std(np_09_2_crash), np.std(np_09_4_crash), np.std(np_09_6_crash)]
    std_dist_09 = [np.std(np_09_1_dist), np.std(np_09_2_dist), np.std(np_09_4_dist), np.std(np_09_6_dist)]
    std_episode_09 = [np.std(np_09_1_episode), np.std(np_09_2_episode), np.std(np_09_4_episode), np.std(np_09_6_episode)]

    return [mean_success_03, mean_success_07, mean_success_09, mean_crash_03, mean_crash_07, mean_crash_09,\
         mean_dist_03, mean_dist_07, mean_dist_09, mean_episode_03, mean_episode_07, mean_episode_09], [std_success_03, std_success_07, \
            std_success_09, std_crash_03, std_crash_07, std_crash_09, std_dist_03, std_dist_07, std_dist_09, std_episode_03, std_episode_07, std_episode_09]



def main():
    filenames_wp = ["results_processing/evaluation_values_1024.txt", "results_processing/evaluation_values_2048.txt", "results_processing/evaluation_values_4096.txt"]
    filenames_np = ["results_processing/evaluation_values_1024_nw.txt", "results_processing/evaluation_values_2048_nw.txt", "results_processing/evaluation_values_4096_nw.txt"]
    best_model = ["results_processing/evaluation_values_4096_nw.txt"]
    curriculum = ["results_processing/evaluation_values.txt"]

    prefix = ["\\textbf{0.3}", "\\textbf{0.7}", '\\textbf{0.9}']

    mean, std = process_evalution(curriculum)
    print("MEAN")
    print("\\\\")
    for i in range(len(mean)):
        for j in range(len(mean[i])):
            # mean[i][j] = str(round(mean[i][j], 3)) + " \\pm " + str(round(std[i][j],3))
            mean[i][j] = str(round(mean[i][j], 3))
        # print(i)
        print(prefix[i%3] + " & " + " & ".join(mean[i]))
        print("\\\\")

    # print("STD")
    # print("\\\\")
    # for j in range(len(std)):
    #     for k in range(len(std[j])):
    #         std[j][k] = str(round(std[j][k],3))
    #     print(prefix[j%3] + " & " + " & ".join(std[j]))
    #     print("\\\\")

if __name__ == '__main__':
    main()
    # run_evaluation()

