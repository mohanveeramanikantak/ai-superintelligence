# Simple Neural Network (1 hidden layer) using NumPy

import numpy as np

# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Dataset (XOR problem)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Initialize weights
np.random.seed(42)
input_layer = 2
hidden_layer = 4
output_layer = 1

W1 = np.random.uniform(size=(input_layer, hidden_layer))
W2 = np.random.uniform(size=(hidden_layer, output_layer))

# Training
for epoch in range(10000):
    # Forward propagation
    hidden_input = np.dot(X, W1)
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2)
    final_output = sigmoid(final_input)

    # Error
    error = y - final_output

    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)
    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update weights
    W2 += hidden_output.T.dot(d_output) * 0.1
    W1 += X.T.dot(d_hidden) * 0.1

# Test
print("Predictions:")
print(final_output)
