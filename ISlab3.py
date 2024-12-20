import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualization

# Step 1: Define the Gaussian Radial Basis Function
def gaussian_rbf(x, c, r):
    """
    Computes the Gaussian RBF for input x.
    :param x: Input values
    :param c: Center of the RBF
    :param r: Radius (spread) of the RBF
    :return: Gaussian RBF output
    """
    return np.exp(-((x - c) ** 2) / (2 * r ** 2))

# Step 2: Generate Training Data
# Define input values (x) spaced evenly between 0.1 and 1.0
x = np.arange(0.1, 1.01, (1-0.1)/40)  # 40 points between 0.1 and 1
# Define the desired output (y) using the provided equation
y = (1 + 0.6 * np.sin((2 * np.pi * x) / 0.7) + 0.3 * np.sin(2 * np.pi * x)) / 2

# Step 3: Define Centers and Radii for RBFs
# These are manually chosen values
c1, r1 = 0.2, 0.15  # Center and radius for the first RBF
c2, r2 = 0.85, 0.15  # Center and radius for the second RBF

# Step 4: Compute RBF Activations for Each Input
phi1 = gaussian_rbf(x, c1, r1)  # Output of the first RBF
phi2 = gaussian_rbf(x, c2, r2)  # Output of the second RBF

# Step 5: Build the Design Matrix
# Stack the two RBF outputs and add a bias term (all ones)
phi = np.vstack((phi1, phi2, np.ones_like(x))).T  # Transpose to shape (N, 3)

# Step 6: Initialize Perceptron Weights
weights = np.random.randn(3)  # Random initial weights for w1, w2, w0
learning_rate = 0.01  # Learning rate for weight updates
epochs = 1000  # Number of training epochs

# Step 7: Train the Network Using Perceptron Learning
for epoch in range(epochs):  # Repeat for the specified number of epochs
    for i in range(len(x)):  # Iterate through each training example
        # Predict the output using the current weights
        y_pred = np.dot(phi[i], weights)
        # Compute the error (difference between desired and predicted output)
        error = y[i] - y_pred
        # Update weights using the perceptron learning rule
        weights += learning_rate * error * phi[i]

# Display the trained weights
print("Trained weights (w1, w2, w0):", weights)

# Step 8: Predict Outputs for All Inputs
y_pred = np.dot(phi, weights)  # Compute predictions for all inputs

# Display the true and predicted outputs
print("True outputs:", y)  # The desired output
print("Predicted outputs:", y_pred)  # The network's output

# Step 9: Plot the Results
plt.figure(figsize=(10, 6))  # Set the figure size
plt.plot(x, y, 'go-', label='Desired Output')  # Plot the true output
plt.plot(x, y_pred, 'r--', label='RBF Network Output')  # Plot the predicted output
plt.xlabel('x')  # Label for x-axis
plt.ylabel('y')  # Label for y-axis
plt.title('RBF Network Approximation')  # Plot title
plt.legend()  # Display legend
plt.grid()  # Add a grid
plt.show()  # Show the plot
