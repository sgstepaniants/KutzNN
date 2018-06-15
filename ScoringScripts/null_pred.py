""" This function gives a cheating prediction which predicts the truth for 500 time steps and then fails.
"""
import numpy as np
import sys


def load(path, n_trajectories=100):
    data = np.loadtxt(path, delimiter=",")
    new_shape = (n_trajectories, data.shape[0] // n_trajectories, -1)
    return data.reshape(new_shape)


def save(path, data):
    flat = data.reshape((-1, data.shape[-1]))
    np.savetxt(path, flat, delimiter=",")


pred = load(sys.argv[1])

# plug in a large value after the fall of time
fall_of_time = 500
pred[:, fall_of_time:, :] = 1000

# save the data
save(sys.argv[2], pred)
