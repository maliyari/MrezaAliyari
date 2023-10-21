# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 20:38:47 2023

@author: ASUS
"""


import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import savemat
from scipy.io import loadmat

# Load the original dataset
train_dataset = h5py.File('C:\\Users\\ASUS\\Downloads\\mreza\\train_catvnoncat.h5', "r")

# Extract the last digit of your Student ID (which is 7)
student_id_last_digit = 7  # Change this to your actual last digit

# Determine the indices for the new training and test sets
total_images = train_dataset['train_set_x'].shape[0]
split_index = 160 + student_id_last_digit
new_train_indices = list(range(160)) + [split_index]
new_test_indices = list(range(160, total_images))
print("New Training Set Indices:", new_train_indices)
print("New Test Set Indices:", new_test_indices)

# Create new training and test sets
new_train_X = np.array(train_dataset["train_set_x"][new_train_indices])
new_train_Y = np.array(train_dataset["train_set_y"][new_train_indices])
new_test_X = np.array(train_dataset["train_set_x"][new_test_indices])
new_test_Y = np.array(train_dataset["train_set_y"][new_test_indices])

# Reshape and normalize the new training set
new_train_X_flat = (new_train_X.reshape(new_train_X.shape[0], -1) / 255).T

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the propagate function
def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = np.sum(((-np.log(A)) * Y + (-np.log(1 - A)) * (1 - Y)))/m
    dw = (np.dot(X, (A - Y).T))/m
    db = (np.sum(A - Y))/m
    cost = np.squeeze(cost)
    return dw, db, cost

# Initialize new training set variables
X = new_train_X_flat
Y = new_train_Y
dim = new_train_X_flat.shape[0]
w = np.zeros((dim, 1))
b = 0
num_iterations = 10000
learning_rate = 0.006
record_cost = 50
costs = []

# Retrain the model
for i in range(num_iterations):
    dw, db, cost = propagate(w, b, X, Y)
    w = w - (learning_rate * dw)
    b = b - (learning_rate * db)
    if i % record_cost == 0:
        costs.append(cost)
        print(f"Cost after iteration {i} is: {cost}")

# Save the new weights and biases
savemat("new_weights.mat", {"new_weights": w})
savemat("new_biases.mat", {"new_biases": b})

# Load the new test set
new_test_X_flat = (new_test_X.reshape(new_test_X.shape[0], -1) / 255).T

# Load the new weights and biases
w = loadmat('new_weights.mat')["new_weights"]
b = loadmat('new_biases.mat')["new_biases"]

# Make predictions on the new test set
A = sigmoid(np.dot(w.T, new_test_X_flat) + b)
Y_predict_test = (A >= 0.5) * 1.0

# Calculate the accuracy on the new training set
Y_predict_train = sigmoid(np.dot(w.T, X) + b)
train_accuracy = 100 - np.mean(np.abs(Y_predict_train - Y)) * 100

# Print the training accuracy
print(f"Training accuracy: {train_accuracy:.2f}%")

# Find misclassified images
misclassified_indices = np.where(Y_predict_test != new_test_Y)[1]
print("Indices of misclassified images:", misclassified_indices)

# Display 4 misclassified images
fig = plt.figure(figsize=(10, 7))
rows = 2
columns = 2
for i in range(4):
    fig.add_subplot(rows, columns, i + 1)
    index = misclassified_indices[i]
    plt.imshow(new_test_X[index])
    plt.axis('off')
    plt.title(f"Predicted: {int(Y_predict_test[index])}, Actual: {int(new_test_Y[index])}")
plt.show()


