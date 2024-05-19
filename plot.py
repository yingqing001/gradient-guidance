# read csv file and plot reward(mean, std) w.r.t steps 

import numpy as np
import matplotlib.pyplot as plt

targets = [2,4,6,8,10]
guidance = 100
guidance = float(guidance)
seed = 5
prompt = 'fox'
colors = ['orange', 'dodgerblue', 'forestgreen', 'tomato', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# create a figure and axis
fig, ax = plt.subplots()
# plot each data-point
x = None
for idx, target in enumerate(targets):
    target = float(target)
    # read csv file
    dir = f'opt/target{target}guidance{guidance}seed{seed}_{prompt}'
    mean_rewards = np.loadtxt(dir + '/mean_rewards.csv', delimiter=',')
    std_rewards = np.loadtxt(dir + '/std_rewards.csv', delimiter=',')
    # set x as discrete number
    x = np.arange(mean_rewards.shape[0], dtype=int)
    ax.plot(x, mean_rewards, color=colors[idx], label=f'target {target}')
    ax.fill_between(x, mean_rewards - std_rewards, mean_rewards + std_rewards, color=colors[idx], alpha=0.1)

# set labels and title
ax.set_xlabel('Optimization Steps')
ax.set_ylabel('Reward')

# save
# Move the legend to the upper left corner, outside the plot area
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.legend(loc='upper right')

# Set the x-axis to use integer steps
plt.xticks(np.arange(min(x), max(x)+1, 5))
plt.savefig(f'guidance{guidance}_seed{seed}_{prompt}.png')
plt.close()
