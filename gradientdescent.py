import numpy as np

# For reproducibility
np.random.seed(42)

# Step 1: Generate Example Data
X1 = np.random.rand(10)
X2 = np.random.rand(10)
Y = 3 + 2 * X1 + 4 * X2 + np.random.randn(10) * 0.5  # Target variable with some noise

# Combine X1 and X2 into a single matrix (including the bias term)
X = np.c_[np.ones(X1.shape[0]), X1, X2]

# Step 2: Define the Gradient Descent Function
def gradient_descent(X, Y, learning_rate=0.01, iterations=1000):
    m, n = X.shape  # m = number of samples, n = number of features including bias
    theta = np.zeros(n)  # Initialize weights to zero
    cost_history = []

    for _ in range(iterations):
        predictions = X.dot(theta)  # Predictions for all samples
        errors = predictions - Y
        gradient = (1/m) * X.T.dot(errors)  # Calculate the gradient
        theta -= learning_rate * gradient  # Update weights

        cost = (1/(2*m)) * np.sum(errors**2)  # Calculate cost (mean squared error)
        cost_history.append(cost)

    return theta, cost_history

# Step 3: Run Gradient Descent
learning_rate = 0.1
iterations = 1000
theta, cost_history = gradient_descent(X, Y, learning_rate, iterations)

print("Estimated coefficients:", theta)

# Step 4: Predict and Evaluate
predictions = X.dot(theta)

print("Predicted values:", predictions)
print("Actual values:", Y)
