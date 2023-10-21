# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 19:58:27 2023

@author: ASUS
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from scipy.io import savemat
from scipy.io import loadmat


train_dataset = h5py.File('C:\\Users\\ASUS\\Downloads\\mreza\\train_catvnoncat.h5', "r")  # change the path
test_dataset = h5py.File('C:\\Users\\ASUS\\Downloads\\mreza\\test_catvnoncat.h5', "r")   # change the path

print(train_dataset.keys())
print(train_dataset['train_set_x']) # x: images of shape (64,64, 3), X contains 209 images (features)
print(train_dataset['train_set_y']) # y: corresponding boolean values (labels)
print(train_dataset['list_classes']) # we have two classes

train_X = np.array(train_dataset["train_set_x"][:])
train_Y = np.array(train_dataset["train_set_y"][:])
test_X = np.array(test_dataset["test_set_x"][:]) #  test set features
test_Y = np.array(test_dataset["test_set_y"][:]) #  test set labels
classes = np.array(test_dataset["list_classes"][:])
print(train_X.shape)
print(train_Y.shape)
print(classes.shape)
print(classes) # numpy.bytes_
train_Y = train_Y.reshape((1, train_Y.shape[0]))
test_Y  = test_Y.reshape((1, test_Y.shape[0]))
print(train_Y.shape)

# the indices of images that we want to look at
indices = [57, 58, 59, 60]

# create figure
fig = plt.figure(figsize=(10, 7))

# setting values to rows and column variables
rows = 2
columns = 2

for i in indices:
  fig.add_subplot(rows, columns, i - 56) # i - 56 is the subplot indices, 1,2,3,4
  plt.imshow(train_X[i])
  plt.axis('off')
  plt.title("y = " + str(train_Y[0, i]) + ", it's a '" + classes[np.squeeze(train_Y[:, i])].decode("utf-8") +  "' picture.")

train_X_flat = (train_X.reshape(train_X.shape[0], -1)/255).T  # flatten the image to have a vector, normalize to prevent the calculations from exploding
print(train_X_flat.shape)
test_X_flat = (test_X.reshape(test_X.shape[0], -1)/255).T

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return: sigmoid(z)
    """
    return 1/(1+np.exp(-z))

def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    cost -- negative log-likelihood cost for logistic regression
    """

    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T,X) + b)
    cost = np.sum(((- np.log(A))*Y + (-np.log(1-A))*(1-Y)))/m  # compute cost

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (np.dot(X,(A-Y).T))/m
    db = (np.sum(A-Y))/m

    cost = np.squeeze(cost)

    return dw, db, cost

X = train_X_flat
Y = train_Y
dim = train_X_flat.shape[0]
w = np.zeros((dim,1)) # initialize w, shape (dim, 1)
b = 0 # initialize b, scalar
num_iterations = 10000
learning_rate = 0.006
record_cost = 50  # print out the cost every 50 iterations
costs = []

for i in range(num_iterations):
    dw, db, cost = propagate(w, b, X, Y)


    # gradient descent
    w = w - (learning_rate*dw)
    b = b - (learning_rate*db)

    # Record the costs
    if i % record_cost == 0:
        costs.append(cost)
        print (f"Cost after iteration {i} is: {cost}")

savemat("weights.mat", {"weights":w})
savemat("biases.mat", {"biases":b})

print(test_X_flat.shape)
w = loadmat('weights.mat')["weights"]
b = loadmat('biases.mat')["biases"]
print(w.shape)
print(b.shape)
A = sigmoid(np.dot(w.T,test_X_flat) + b)
Y_predict_test = (A >= 0.5) * 1.0
Y_predict_train = sigmoid(np.dot(w.T,X) + b)

print(f"train accuracy: {(100 - np.mean(np.abs(Y_predict_train - train_Y)) * 100):2f}")
print(f"test accuracy: {(100 - np.mean(np.abs(Y_predict_test - test_Y)) * 100):2f}")


