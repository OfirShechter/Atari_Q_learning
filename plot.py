import pickle
import matplotlib.pyplot as plt

file = 'statistics_1622106577.3121212.pkl'
with open(file, 'rb') as f:
    stats = pickle.load(f)

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plt.plot([100 * (1 + i) for i in range(len(stats['mean_episode_rewards']))], stats['mean_episode_rewards'])
plt.plot([100 * (1 + i) for i in range(len(stats['best_mean_episode_rewards']))], stats['best_mean_episode_rewards'])

plt.legend(['Mean Reward', 'Best Mean Reward'])

plt.show()
