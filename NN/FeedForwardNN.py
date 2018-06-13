from scipy.io import loadmat
import numpy as np
import random
import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error

# NEURAL NET WITH NO BURN-IN (ONLY TAKES 1 POINT ON INPUT)

#******************************************************************************
# Read Data
#******************************************************************************
num_traj = 100
tmax = 1500

X = loadmat('../Data/TestData.mat')
# X_input is 100 (x, y, z) trajectories of particles on the Lorentz attractor from time 1 to tmax-1
X_input = X['input']
# X_output is 100 (x, y, z) trajectories of the same particles on the Lorentz attractor from time 2 to tmax
X_output = X['output']
# Note that X_input and X_output are data from the same particles so they overlap for tmax-2 time points

X_noisy = loadmat('../Data/TestData2.mat')
# X_noisy_input is X_input with noise added on top
X_noisy_input = X_noisy['input2']
# X_noisy_output is X_output with noise added on top
X_noisy_output = X_noisy['output2']


#******************************************************************************
# Rescale data between 0 and 1 for learning
#******************************************************************************
Xmin = X_input.min()
Xmax = X_input.max()
    
X_input = ((X_input - Xmin) / (Xmax - Xmin)) 
X_output = ((X_output - Xmin) / (Xmax - Xmin))


#******************************************************************************
# Preprocess Data
#******************************************************************************
d = X_input.shape[1]
train_size = 90000

# train the neural net on the first train_size time points and test its predictions on the rest
X_train = torch.Tensor(X_input[0:train_size])
Y_train = torch.Tensor(X_output[0:train_size])
X_test = torch.Tensor(X_input[train_size:])
Y_test = torch.Tensor(X_output[train_size:])

X_noisy_train = torch.Tensor(X_noisy_input[0:train_size])
Y_noisy_train = torch.Tensor(X_noisy_output[0:train_size])
X_noisy_test = torch.Tensor(X_noisy_input[train_size:])
Y_noisy_test = torch.Tensor(X_noisy_output[train_size:])


#******************************************************************************
# Build Feed-Forward Neural Network
#******************************************************************************
# 2 hidden layers
d1 = 10
d2 = 100
# build the computational graph
network_model = torch.nn.Sequential(
                torch.nn.Linear(d, d1),
                torch.nn.ReLU(True),
                torch.nn.Linear(d1, d2),
                torch.nn.ReLU(True),
                torch.nn.Linear(d2, d)
            )


#******************************************************************************
# Train Feed-Forward Neural Network
#******************************************************************************
# store train and test squared errors for each iteration
train_err = []
test_err = []

# number of epochs/iterations for training the neural net
num_epochs = 1000
batch_size = 1000
learning_rate = 0.01
weight_decay = 1e-5


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
        learning_rate *= 0.8
        weight_decay *= 0.8        
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

print "Neural Net Min Train MSE: " + str(min(train_err))
print "Neural Net Min Test MSE: " + str(min(test_err))


#******************************************************************************
# Apply Neural Net Several Times to Predict Future Points
#******************************************************************************
Y_true = X_output  # the true future evolution of all particles (after the first point)
Y_pred = np.zeros((num_traj * tmax, d))  # the predicted future evolution of all particles (after the first point)
for i in range(num_traj):
    x_t = X_input[i*tmax]
    for j in range(tmax):
        x_t = network_model(Variable(torch.Tensor(x_t))).data.numpy()
        Y_pred[i * tmax + j] = x_t

print "Model MSE: " + str(mean_squared_error(Y_true, Y_pred))


#******************************************************************************
# Plot True and Model Predicted Trajectories On Test Data (no burn-in)
#******************************************************************************
# trajectory number to plot (100 trajectories in total)
traj = 4
init_point = X_input[tmax*traj]
true_traj = Y_true[tmax*traj:tmax*(traj+1)]
pred_traj = Y_pred[tmax*traj:tmax*(traj+1)]

ax = plt.axes(projection='3d')

# plot start 'o' and end '^' points of trajectories
ax.scatter(init_point[0], init_point[1], init_point[2], s=50, c='g', marker='o')
ax.scatter(true_traj[0, 0], true_traj[0, 1], true_traj[0, 2], s=50, c='b', marker='o')
ax.scatter(true_traj[-1, 0], true_traj[-1, 1], true_traj[-1, 2], s=50, c='b', marker='^')
ax.scatter(pred_traj[0, 0], pred_traj[0, 1], pred_traj[0, 2], s=50, c='r', marker='o')
ax.scatter(pred_traj[-1, 0], pred_traj[-1, 1], pred_traj[-1, 2], s=50, c='r', marker='^')

# plot true, and predicted trajectories
ax.plot3D(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2], c='b', label='true')
ax.plot3D(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], c='r', label='pred')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.show()
