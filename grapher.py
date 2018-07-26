import matplotlib.pyplot as plt;
from runner import graphfile_format
from runner import environments
import sys
import numpy as np
import pickle

graph_imgs_format = "reports/graphs/{0}.pgf"

for env in environments:
    graph_datas = []

    for run in range(3):
        graphfile = graphfile_format.format(env, run)
        with open(graphfile, 'rb') as file:
            graph_datas.append(pickle.load(file))

    graph_data = [np.mean(np.array(d), axis=0) for d in list(zip(*graph_datas))]

    timesteps, min_rews, max_rews, avg_rews = graph_data;
    plt.figure()

    plt.ylabel("Number of Timesteps")
    plt.xlabel("Reward")
    plt.title("{0} PPO Results".format(env))

    plt.plot(timesteps, avg_rews, linestyle='-', color=(0.0, 0.0, 0.0))
    plt.plot(timesteps, min_rews, linestyle='-', color=(0.5, 0.5, 0.5))
    plt.plot(timesteps, max_rews, linestyle='-', color=(0.5, 0.5, 0.5))

    plt.savefig(graph_imgs_format.format(env))

#plt.show()
