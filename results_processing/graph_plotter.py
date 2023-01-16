from tbparser.summary_reader import SummaryReader
import numpy as np
import matplotlib.pyplot as plt

logdir_4096 = "/Users/emilymorris/ray_results/SAC_SimpleRobotEnvironmentWP_2023-01-11_16-00-50k1ew5bbn"
logdir_2048 = "/Users/emilymorris/ray_results/SAC_SimpleRobotEnvironmentWP_2023-01-11_15-17-1741kday90"
logdir_1024 = "/Users/emilymorris/ray_results/SAC_SimpleRobotEnvironmentWP_2023-01-13_17-03-57wu5205lh"

# logdir_4096 = "/Users/emilymorris/ray_results/SAC_SimpleRobotEnviromentCO_2023-01-11_11-48-18_3vmn6e3"
# logdir_2048 = "/Users/emilymorris/ray_results/SAC_SimpleRobotEnviromentCO_2023-01-10_23-12-41qecbjmyh"
# logdir_1024 = "/Users/emilymorris/ray_results/SAC_SimpleRobotEnviromentCO_2023-01-11_09-15-29l59naez1"

curriculum_half = "/Users/emilymorris/ray_results/SAC_CurriculumSimpleEnvironment_2023-01-12_12-06-33weavy2gh"
curriculum_001 = "/Users/emilymorris/ray_results/SAC_CurriculumSimpleEnvironment_2023-01-13_10-09-40czkn_fcg"

# reader_4096 = SummaryReader(logdir_4096, tag_filter=["ray/tune/custom_metrics/reached_goal_mean"])
# reader_2048 = SummaryReader(logdir_2048, tag_filter=["ray/tune/custom_metrics/reached_goal_mean"])
# reader_1024 = SummaryReader(logdir_1024, tag_filter=["ray/tune/custom_metrics/reached_goal_mean"])

reader_half = SummaryReader(curriculum_half, tag_filter=["ray/tune/custom_metrics/reached_goal_mean"])
reader_001 = SummaryReader(curriculum_001, tag_filter=["ray/tune/custom_metrics/reached_goal_mean"])


def extract_vals(summaryReader:SummaryReader):
    x_vals = []
    y_vals = []
    for val in summaryReader.__iter__():
        y_vals.append(val.step)
        x_vals.append(val.value)
    return y_vals, x_vals

# y_vals_4096, x_vals_4096 = extract_vals(reader_4096)
# y_vals_2048, x_vals_2048 = extract_vals(reader_2048)
# y_vals_1024, x_vals_1024 = extract_vals(reader_1024)
# print(len(y_vals_1024), len(x_vals_1024), len(y_vals_2048), len(x_vals_2048), len(y_vals_4096), len(x_vals_4096))

y_vals_half, x_vals_half = extract_vals(reader_half)
y_vals_001, x_vals_001 = extract_vals(reader_001)

# min_length = min(len(y_vals_1024), len(y_vals_2048), len(y_vals_4096))
# stacks_vals = [x_vals_2048[:min_length], x_vals_1024[:min_length], x_vals_4096[:min_length]]
# std = np.std(stacks_vals, axis=0)
# mean = np.mean(stacks_vals, axis=0)
# print(len(std), len(mean))

plt.ticklabel_format(scilimits=(0,0))
plt.xlabel('Steps')
plt.ylabel('Average success')

## Regular graph plotting
# plt.plot(y_vals_2048, x_vals_2048[:min_length], color='red')
# plt.plot(y_vals_2048, x_vals_1024[:min_length], color='blue')
# plt.plot(y_vals_2048, x_vals_4096[:min_length], color='green')
# plt.plot(y_vals_2048, mean, color='darkorange')
# plt.fill_between(y_vals_2048, mean + std, mean - std, color='bisque', alpha=.5)

## Curriculum learning plotting 
# First change {'goal_tolerance': 0.08, 'tolerance_step_change': 502736}
    # Second change {'goal_tolerance': 0.04, 'tolerance_step_change': 680000} (I think)
    # Third change {'goal_tolerance': 0.02, 'tolerance_step_change': 1484720}
y_vals_switch = np.linspace(0.0, 1.0, num=50)
tolerance_008 = np.full((50), 502736)
tolerance_004 = np.full((50), 680000)
tolerance_002 = np.full((50), 1484720)
tolerance_003 = np.full((50), 1437400)
tolerance_0021 = np.full((50), 2241920)
plt.plot(y_vals_half, x_vals_half, color='darkorange', label="Halving")
plt.plot(y_vals_001, x_vals_001, color='cornflowerblue', label="Minusing")
plt.plot(tolerance_008, y_vals_switch, color='black', linestyle='dashed', label="Tolerance H")
plt.plot(tolerance_004, y_vals_switch, color='black', linestyle='dashed')
plt.plot(tolerance_002, y_vals_switch, color='black', linestyle='dashed')
plt.plot(tolerance_003, y_vals_switch, color='grey', linestyle='dashed', label="Tolerance M")
plt.plot(tolerance_0021, y_vals_switch, color='grey', linestyle='dashed')
plt.legend()
plt.show()



# print(y_vals_1024)
# print(y_vals_2048)
# print(y_vals_4096)
