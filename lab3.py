import numpy as np
import matplotlib.pyplot as plt

def gaussian_rbf(x, c, r):
    return np.exp(-((x - c) ** 2) / (2 * r ** 2))

x = np.arange(0.1, 1.01, (1-0.1)/40)
y = (1 + 0.6 * np.sin((2 * np.pi * x) / 0.7) + 0.3 * np.sin(2 * np.pi * x)) / 2


#Cetre and radius 
c1, r1 = 0.2, 0.15
c2, r2 = 0.85, 0.15

#outputs of two Gaussian radial basis
phi1 = gaussian_rbf(x, c1, r1)
phi2 = gaussian_rbf(x, c2, r2)
# Adding a bias term to the activations
phi = np.vstack((phi1, phi2, np.ones_like(x))).T  # Design matrix with bias

weights = np.random.randn(3) #Generating 3 weights
learning_rate = 0.01
epochs = 1000

# Perceptron training algorithm
for epoch in range(epochs):
    for i in range(len(x)):
        # Predicting output
        y_pred = np.dot(phi[i], weights)
        # Computing error
        error = y[i] - y_pred
        # Update weights
        weights += learning_rate * error * phi[i]

# Displaying the weights
print("Trained weights (w1, w2, w0):", weights)

# Predicted output
y_pred = np.dot(phi, weights)

print("True outputs:", y)
print("Predicted outputs:", y_pred)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'bo-', label='Desired Output',color='g')
plt.plot(x, y_pred, 'r--', label='RBF Network Output')
plt.xlabel('x')
plt.ylabel('y')
plt.title('RBF Network Approximation')
plt.legend()
plt.grid()
plt.show()