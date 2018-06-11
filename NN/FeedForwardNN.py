from scipy.io import loadmat
import numpy as np

X = loadmat('../Data/ContestData.mat')
# X_train is the (x, y, z) position of the particle on the Lorentz attractor from time 1 to m-1
X_train = X['input']
# X_test is the (x, y, z) position of the same particle on the Lorentz attractor from time 2 to m
X_test = X['output']
# Note that X_train and X_test are data from the same particle so they overlap for m-2 time points

X_noisy = loadmat('../Data/ContestData2.mat')
# X_noisy_train is X_train with noise added on top
X_noisy_train = X_noisy['input2']
# X_noisy_test is X_test with noise added on top
X_noisy_test = X_noisy['output2']
