import numpy as np
import matplotlib.pyplot as plt

# Generate some example data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
Y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relationship with some noise

# Add x0 = 1 to each instance (bias term)
X_b = np.c_[np.ones((100, 1)), X]

# Gradient Descent Function
def gradient_descent(X, Y, learning_rate=0.1, iterations=1000):
    m = len(Y)
    theta = np.random.randn(2, 1)  # Random initialization of theta
    for iteration in range(iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - Y)
        theta = theta - learning_rate * gradients
    return theta

# Run Gradient Descent
theta_best = gradient_descent(X_b, Y)
print("Estimated coefficients:", theta_best)

# Make predictions
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # Add bias term
Y_predict = X_new_b.dot(theta_best)

print("Predicted values:", Y_predict)

# Plot the results
plt.plot(X_new, Y_predict, "r-", label="Predictions")
plt.plot(X, Y, "b.", label="Training data")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
