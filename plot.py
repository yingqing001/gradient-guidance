# read csv file and plot reward(mean, std) w.r.t steps 

import numpy as np
import matplotlib.pyplot as plt

targets = [1,2,4,6]
guidance = 100
guidance = float(guidance)
seed = 5
prompt = 'fox'
colors = ['orange', 'dodgerblue', 'forestgreen', 'tomato', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# create a figure and axis
fig, ax = plt.subplots()
# plot each data-point
for idx, target in enumerate(targets):
    target = float(target)
    # read csv file
    dir = f'opt/target{target}guidance{guidance}seed{seed}_{prompt}'
    mean_rewards = np.loadtxt(dir + '/mean_rewards.csv', delimiter=',')
    std_rewards = np.loadtxt(dir + '/std_rewards.csv', delimiter=',')
    x = np.arange(mean_rewards.shape[0])
    ax.plot(x, mean_rewards, color=colors[idx], label=f'target {target}')
    #ax.fill_between(x, mean_rewards - std_rewards, mean_rewards + std_rewards, color=colors[idx], alpha=0.1)

# set labels and title
ax.set_xlabel('Optimization Steps')
ax.set_ylabel('Reward')

# save
plt.legend()
plt.savefig(f'guidance{guidance}_seed{seed}_{prompt}.png')
plt.close()
