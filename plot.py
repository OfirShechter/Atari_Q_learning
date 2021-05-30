import pickle
import matplotlib.pyplot as plt

file = 'statistics_1622106577.3121212.pkl'
file2 = 'statistics_1622105417.228623.pkl'
file3 = 'statistics_1622104529.6831777.pkl'
file4 = 'statistics_1622101925.023222.pkl'

def Q1():
    with open(file, 'rb') as f:
        stats = pickle.load(f)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.plot([100 * (1 + i) for i in range(len(stats['mean_episode_rewards']))], stats['mean_episode_rewards'])
    plt.plot([100 * (1 + i) for i in range(len(stats['best_mean_episode_rewards']))],
             stats['best_mean_episode_rewards'])
    plt.legend(['Mean Reward', 'Best Mean Reward'])
    plt.show()

def Q1():
    file = 'statistics_1622106577.3121212.pkl'
    with open(file, 'rb') as f:
        stats = pickle.load(f)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.plot([100 * (1 + i) for i in range(len(stats['mean_episode_rewards']))], stats['mean_episode_rewards'])
    plt.plot([100 * (1 + i) for i in range(len(stats['best_mean_episode_rewards']))],
             stats['best_mean_episode_rewards'])
    plt.legend(['Mean Reward', 'Best Mean Reward'])
    plt.show()


Q1()


