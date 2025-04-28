import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Manual computation
X_flat = X.flatten()
y_flat = y.flatten()

# Calculate means
x_mean = np.mean(X_flat)
y_mean = np.mean(y_flat)

# Calculate theta1 and theta0
theta1 = np.sum((X_flat - x_mean) * (y_flat - y_mean)) / np.sum((X_flat - x_mean)**2)
theta0 = y_mean - theta1 * x_mean

print(f"Manual Linear Regression Parameters:")
print(f"Intercept (theta0): {theta0:.2f}")
print(f"Coefficient (theta1): {theta1:.2f}")

# Predict
y_pred = theta0 + theta1 * X_flat

# Plot
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.title("Manual Linear Regression Fit")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
