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


#******************************************************************************
# Read Data
#******************************************************************************
X = loadmat('../Data/ContestData.mat')
# X_input is the (x, y, z) position of the particle on the Lorentz attractor from time 1 to tmax-1
X_input = X['input']
# X_output is the (x, y, z) position of the same particle on the Lorentz attractor from time 2 to tmax
X_output = X['output']
# Note that X_input and X_output are data from the same particle so they overlap for tmax-2 time points

X_noisy = loadmat('../Data/ContestData2.mat')
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
tmax, d = X_input.shape
# define the burn-in time
train_size = 15000

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
# 100 hidden layers
d1 = 10
d2 = 10
# build the computational graph
network_model = torch.nn.Sequential(
                torch.nn.Linear(d, d1),
                torch.nn.ReLU(True),
                torch.nn.Linear(d1, d2),
                torch.nn.ReLU(True),
                torch.nn.Linear(d2, d),
                torch.nn.Sigmoid(),
            )


#******************************************************************************
# Train Feed-Forward Neural Network
#******************************************************************************
# store train and test squared errors for each iteration
train_err = []
test_err = []

# number of epochs/iterations for training the neural net
num_epochs = 1000
batch_size = 200
learning_rate = 0.01
weight_decay  = 1e-5


opt = optim.Adam(network_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# for each epoch, do a step of SGD and recompute the weights of the neural net
freq = 5
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
    if idx % 100 == 0:

        learning_rate *= 0.8
        weight_decay *= 0.8        
        opt = optim.Adam(network_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            
        


#******************************************************************************
# Plot Train and Test Errors
#******************************************************************************
xs = np.arange(0, num_epochs / freq)
plt.title('Mean Squared Error over Effective Epochs')
plt.plot(xs, train_err, 'bo', ms=2, label='train')
plt.plot(xs, test_err, 'ro', ms=2, label='test')
plt.legend(loc='upper right', title='Legend')
plt.xlabel('Effective Epoch')
plt.ylabel('Mean Squared Error')
plt.show()


#******************************************************************************
# Plot True and Predicted Trajectories On Test Data (after burn-in)
#******************************************************************************
# time to plot true and prediction trajectories out to
tpred = 1000
X_true = X_train.numpy()
Y_true = Y_test.numpy()[0:tpred]
Y_pred = network_model(Variable(X_test)).data.numpy()[0:tpred]

ax = plt.axes(projection='3d')

# plot start 'o' and end '^' points of trajectories
ax.scatter(X_true[0, 0], X_true[0, 1], X_true[0, 2], s=50, c='g', marker='o')
ax.scatter(X_true[-1, 0], X_true[-1, 1], X_true[-1, 2], s=50, c='g', marker='^')
ax.scatter(Y_true[0, 0], Y_true[0, 1], Y_true[0, 2], s=50, c='b', marker='o')
ax.scatter(Y_true[-1, 0], Y_true[-1, 1], Y_true[-1, 2], s=50, c='b', marker='^')
ax.scatter(Y_pred[0, 0], Y_pred[0, 1], Y_pred[0, 2], s=50, c='r', marker='o')
ax.scatter(Y_pred[-1, 0], Y_pred[-1, 1], Y_pred[-1, 2], s=50, c='r', marker='^')

# plot burn-in, true, and predicted trajectories
ax.plot3D(X_true[:, 0], X_true[:, 1], X_true[:, 2], c='g', label='burn-in')
ax.plot3D(Y_true[:, 0], Y_true[:, 1], Y_true[:, 2], c='b', label='true')
ax.plot3D(Y_pred[:, 0], Y_pred[:, 1], Y_pred[:, 2], c='r', label='pred')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.show()
