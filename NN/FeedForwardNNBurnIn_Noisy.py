from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.magic('reset -sf')

from scipy.io import loadmat
import os
import numpy as np
import pandas as pd
import random
import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error

# NEURAL NET WITH 50 POINTS FOR BURN-IN WITH NOISY DATA (challenge b)
challenge = 'b'

#******************************************************************************
# Read Data
#******************************************************************************
num_traj = 100
tmax = 1500

X = loadmat('../Data/TestData2.mat')
# X_input is 100 (x, y, z) trajectories of particles on the Lorentz attractor from time 1 to tmax-1
X_input = X['input2']
# X_output is 100 (x, y, z) trajectories of the same particles on the Lorentz attractor from time 2 to tmax
X_output = X['output2']


#******************************************************************************
# Rescale data between 0 and 1 for learning
#******************************************************************************
#Xmin = X_input.min()
#Xmax = X_input.max()

#X_input = ((X_input - Xmin) / (Xmax - Xmin))
#X_output = ((X_output - Xmin) / (Xmax - Xmin))


#******************************************************************************
# Preprocess Data
#******************************************************************************
d = X_input.shape[1]
burn_in = 50
train_size = 90000

# format the non-noisy data for the neural net
X = np.zeros((num_traj * (tmax - burn_in + 1), burn_in * d))
Y = np.zeros((num_traj * (tmax - burn_in + 1), d))

for i in range(num_traj):
    for j in range(tmax-burn_in+1):
        # get previous burn_in time points
        x = X_input[tmax*i+j:tmax*i+j+burn_in].flatten()
        # get the next time point
        y = X_output[tmax*i+j+burn_in-1]
        X[(tmax-burn_in+1)*i+j] = x
        Y[(tmax-burn_in+1)*i+j] = y

# shuffle the data
idx = random.sample(range(0, num_traj * (tmax - burn_in + 1)), num_traj * (tmax - burn_in + 1))
X_shuffled = X[idx]
Y_shuffled = Y[idx]

# split data into test and train sets
X_train = torch.Tensor(X_shuffled[0:train_size])
Y_train = torch.Tensor(Y_shuffled[0:train_size])
X_test = torch.Tensor(X_shuffled[train_size:])
Y_test = torch.Tensor(Y_shuffled[train_size:])


#******************************************************************************
# Build Feed-Forward Neural Network
#******************************************************************************
# 2 hidden layers
di = burn_in * d
d1 = 100
d2 = 1000
df = d
# build the computational graph
network_model = torch.nn.Sequential(
                torch.nn.Linear(di, d1),
                torch.nn.ReLU(True),
                torch.nn.Linear(d1, d2),
                torch.nn.ReLU(True),
                torch.nn.Linear(d2, df),
            )

#******************************************************************************
# Train Feed-Forward Neural Network
#******************************************************************************
# store train and test squared errors for each iteration
train_err = []
test_err = []

# number of epochs/iterations for training the neural net
num_epochs = 15000
batch_size = 1000
learning_rate = 0.01
weight_decay = 1e-4


opt = optim.Adam(network_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# for each epoch, do a step of SGD and recompute the weights of the neural net
freq = 100
for idx in range(num_epochs):
    print(idx)
    # save the train and test errors every time we have sampled freq batches
    if idx % freq == 0:
        Y_train_hat = network_model(Variable(X_train)).data
        train_err.append(mean_squared_error(Y_train, Y_train_hat))
        Y_test_hat = network_model(Variable(X_test)).data
        test_err.append(mean_squared_error(Y_test, Y_test_hat))

    # train the neural net using mini-batch SGD
    batch_idx = random.sample(range(0, train_size), batch_size)
    network_model.zero_grad()

    # get a batch sample of the output
    Y_batch_train = Variable(Y_train[batch_idx, :])
    # predict the batch sample of the output
    Y_batch_train_hat = network_model(Variable(X_train[batch_idx, :]))
    # compute the mean squared error as our loss
    loss = F.mse_loss(Y_batch_train, Y_batch_train_hat)

    # this computes the gradient for us!
    loss.backward()
    # this does the parameter for us!
    opt.step()


    # ===================adjusted lr========================
    if idx % freq == 0:
        if idx < 2000:
            learning_rate *= 0.8
            weight_decay *= 0.5
            opt = optim.Adam(network_model.parameters(), lr=learning_rate, weight_decay=weight_decay)


#******************************************************************************
# Plot Train and Test Errors for Neural Net
#******************************************************************************
xs = np.arange(0, num_epochs / freq)
plt.title('Neural Net Mean Squared Error over Effective Epochs')
plt.plot(xs, train_err, 'bo', ms=2, label='train')
plt.plot(xs, test_err, 'ro', ms=2, label='test')
plt.legend(loc='upper right', title='Legend')
plt.xlabel('Effective Epoch')
plt.ylabel('Mean Squared Error')
plt.show()

output_text = "Neural Net Min Train MSE: " + str(min(train_err))
print(output_text)
output_text = "Neural Net Min Test MSE: " + str(min(test_err))
print(output_text)


#******************************************************************************
# Apply Neural Net Several Times to Predict Future Points (ON TRAIN DATA)
#******************************************************************************
# the true future evolution of all particles (after the first burn_in points)
Y_true = np.empty_like(X_input)
Y_true[:] = X_input
# the predicted future evolution of all particles (after the first burn_in points)
Y_pred = np.empty_like(X_input)
Y_pred[:] = X_input

for i in range(num_traj):
    for j in range(tmax-burn_in):
        # get previous burn_in time points
        xs = Y_pred[tmax*i+j:tmax*i+j+burn_in].flatten()
        # get the next time point from the neural net
        y = network_model(Variable(torch.Tensor(xs))).data.numpy()
        Y_pred[tmax*i+burn_in+j] = y
        print tmax*i+burn_in+j

output_text = "Model MSE: " + str(mean_squared_error(Y_true, Y_pred))
print(output_text)

output_text = "Model R2: " + str(np.corrcoef(Y_true.flatten(), Y_pred.flatten())[0, 1])
print(output_text)


#******************************************************************************
# Plot True and Model Predicted Trajectories (ON TRAIN DATA)
#******************************************************************************
# trajectory number to plot (100 trajectories in total)
traj = 0
burn_in_traj = X_input[tmax*traj:tmax*traj+burn_in]
true_traj = Y_true[traj*tmax:(traj+1)*tmax]
pred_traj = Y_pred[traj*tmax:(traj+1)*tmax]

ax = plt.axes(projection='3d')

# plot start 'o' and end '^' points of trajectories
ax.scatter(burn_in_traj[0, 0], burn_in_traj[0, 1], burn_in_traj[0, 2], s=50, c='g', marker='o')
ax.scatter(burn_in_traj[-1, 0], burn_in_traj[-1, 1], burn_in_traj[-1, 2], s=50, c='g', marker='^')
ax.scatter(true_traj[-1, 0], true_traj[-1, 1], true_traj[-1, 2], s=50, c='b', marker='^')
ax.scatter(pred_traj[-1, 0], pred_traj[-1, 1], pred_traj[-1, 2], s=50, c='r', marker='^')

# plot true, and predicted trajectories
ax.plot3D(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2], c='b', label='true')
ax.plot3D(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], c='r', label='pred')
ax.plot3D(burn_in_traj[:, 0], burn_in_traj[:, 1], burn_in_traj[:, 2], c='g', label='burn in')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.show()


#******************************************************************************
# Apply Neural Net Several Times to Predict Future Points (ON TEST DATA)
#******************************************************************************
df = pd.read_csv('../Data/' + challenge + '_test.csv', sep=',',header=None)
X_input_test = df.values
# the true future evolution of all particles (after the first burn_in points)
Y_true_test = np.empty_like(X_input)
Y_true_test[:] = X_input
# the predicted future evolution of all particles (after the first burn_in points)
Y_pred_test = np.empty_like(X_input)
Y_pred_test[:] = X_input

for i in range(num_traj):
    for j in range(tmax-burn_in):
        # get previous burn_in time points
        xs = Y_pred_test[tmax*i+j:tmax*i+j+burn_in].flatten()
        # get the next time point from the neural net
        y = network_model(Variable(torch.Tensor(xs))).data.numpy()
        Y_pred_test[tmax*i+burn_in+j] = y
        print tmax*i+burn_in+j

output_text = "Model MSE: " + str(mean_squared_error(Y_true_test, Y_pred_test))
print(output_text)

output_text = "Model R2: " + str(np.corrcoef(Y_true_test, Y_pred_test)[0, 1])
print(output_text)


#******************************************************************************
# Plot True and Model Predicted Trajectories (ON TEST DATA)
#******************************************************************************
# trajectory number to plot (100 trajectories in total)
traj = 0
burn_in_traj = X_input_test[tmax*traj:tmax*traj+burn_in]
true_traj = Y_true_test[traj*tmax:(traj+1)*tmax]
pred_traj = Y_pred_test[traj*tmax:(traj+1)*tmax]

ax = plt.axes(projection='3d')

# plot start 'o' and end '^' points of trajectories
ax.scatter(burn_in_traj[0, 0], burn_in_traj[0, 1], burn_in_traj[0, 2], s=50, c='g', marker='o')
ax.scatter(burn_in_traj[-1, 0], burn_in_traj[-1, 1], burn_in_traj[-1, 2], s=50, c='g', marker='^')
ax.scatter(true_traj[-1, 0], true_traj[-1, 1], true_traj[-1, 2], s=50, c='b', marker='^')
ax.scatter(pred_traj[-1, 0], pred_traj[-1, 1], pred_traj[-1, 2], s=50, c='r', marker='^')

# plot true, and predicted trajectories
ax.plot3D(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2], c='b', label='true')
ax.plot3D(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], c='r', label='pred')
ax.plot3D(burn_in_traj[:, 0], burn_in_traj[:, 1], burn_in_traj[:, 2], c='g', label='burn in')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.show()


#******************************************************************************
# Output CSV File of Predicted Trajectories (ON TEST DATA)
#******************************************************************************
filename = '../PowerRangers/' + challenge + '_burn_in.csv'
if not os.path.exists(os.path.dirname(filename)):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
np.savetxt(filename, Y_pred, delimiter=",")
