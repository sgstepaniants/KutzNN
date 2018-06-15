#!/usr/bin/env python
"""
3.  Remember our challenge NN problem.  I will be putting up “training data” soon on the Lorenz attractor.   Objectives
	(a) How well can you train the network to predict trajectories and not fall off for a test set I give  later on
	(b) How well can you do (a) with noise training/test data
	(c) How well can you predict the the Lorenz when you only have measurements of the x variable
	(d) How well can you do this for noise training/test data for x
	(e) Given test/training data across a bunch of values of rho, how well can you train a NN to stick on trajectories for a test set with different rho values


Evaluation metrics:

1. Prediction Error: RMS
2. Attractor Errors: covariance, autocorrelation function

"""
import matplotlib.pyplot as plt
import sys
import numpy as np


def load(path, n_trajectories=100):
    data = np.loadtxt(path, delimiter=",")
    new_shape = (n_trajectories, data.shape[0] // n_trajectories, -1)
    return data.reshape(new_shape)


def rms(x):
    return np.sqrt(((x)**2).mean(axis=(0, -1)))


def rms_error(truth, pred):
    return rms(truth - pred)


def falloff_time(truth, pred, threshold=7):
    if truth.shape != pred.shape:
        raise ValueError(f"The shape of pred is {pred.shape} but truth has "
                         f"shape {truth.shape}")

    rms = rms_error(truth, pred)
    for i in range(rms.shape[0]):
        if rms[i] > threshold:
            break
    return i


def auc(truth, pred):
    return np.sqrt(((truth-pred)**2).mean())


def main():
    truth = sys.argv[1]
    pred = sys.argv[2]

    truth = load(truth)
    pred = load(pred)

    scores = [
        falloff_time(truth, pred),
        auc(truth, pred)
    ]

    print('%d,%.4f'%(scores[0], scores[1]))


if __name__ == '__main__':
    main()
