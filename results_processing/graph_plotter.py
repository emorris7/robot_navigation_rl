from tbparser.summary_reader import SummaryReader
import numpy as np
import matplotlib.pyplot as plt

# logdir_4096 = "/Users/emilymorris/ray_results/SAC_SimpleRobotEnvironmentWP_2023-01-11_16-00-50k1ew5bbn"
# logdir_2048 = "/Users/emilymorris/ray_results/SAC_SimpleRobotEnvironmentWP_2023-01-11_15-17-1741kday90"
# logdir_1024 = "/Users/emilymorris/ray_results/SAC_SimpleRobotEnvironmentWP_2023-01-11_13-36-01w7hn_rla"

logdir_4096 = "/Users/emilymorris/ray_results/SAC_SimpleRobotEnviromentCO_2023-01-11_11-48-18_3vmn6e3"
logdir_2048 = "/Users/emilymorris/ray_results/SAC_SimpleRobotEnviromentCO_2023-01-10_23-12-41qecbjmyh"
logdir_1024 = "/Users/emilymorris/ray_results/SAC_SimpleRobotEnviromentCO_2023-01-11_09-15-29l59naez1"

reader_4096 = SummaryReader(logdir_4096, tag_filter=["ray/tune/custom_metrics/reached_goal_mean"])
reader_2048 = SummaryReader(logdir_2048, tag_filter=["ray/tune/custom_metrics/reached_goal_mean"])
reader_1024 = SummaryReader(logdir_1024, tag_filter=["ray/tune/custom_metrics/reached_goal_mean"])

def extract_vals(summaryReader:SummaryReader):
    x_vals = []
    y_vals = []
    for val in summaryReader.__iter__():
        y_vals.append(val.step)
        x_vals.append(val.value)
    return y_vals, x_vals

y_vals_4096, x_vals_4096 = extract_vals(reader_4096)
y_vals_2048, x_vals_2048 = extract_vals(reader_2048)
y_vals_1024, x_vals_1024 = extract_vals(reader_1024)
print(len(y_vals_1024), len(x_vals_1024), len(y_vals_2048), len(x_vals_2048), len(y_vals_4096), len(x_vals_4096))

min_length = min(len(y_vals_1024), len(y_vals_2048), len(y_vals_4096))
stacks_vals = [x_vals_2048[:min_length], x_vals_1024[:min_length], x_vals_4096[:min_length]]
std = np.std(stacks_vals, axis=0)
mean = np.mean(stacks_vals, axis=0)
print(len(std), len(mean))

plt.ticklabel_format(scilimits=(0,0))
plt.xlabel('Steps')
plt.ylabel('Average success')
plt.plot(y_vals_4096, mean, color='darkorange')
plt.fill_between(y_vals_4096, mean + std, mean - std, color='bisque', alpha=.5)
plt.show()



# print(y_vals_1024)
# print(y_vals_2048)
# print(y_vals_4096)
