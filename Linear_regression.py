# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate some synthetic data
np.random.seed(42)
height = np.random.uniform(low=150, high=190, size=100)
weight = 0.6 * height + 45 + np.random.normal(scale=10, size=100)

# Reshape the data
height = height.reshape(-1, 1)  # Reshape to a column vector

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(height, weight)

# Make predictions for new data points
new_heights = np.array([160, 170, 180]).reshape(-1, 1)
predicted_weights = model.predict(new_heights)

# Plot the data and the linear regression line
plt.scatter(height, weight, label='Original Data')
plt.plot(height, model.predict(height), color='red', label='Linear Regression Line')
plt.scatter(new_heights, predicted_weights, color='green', marker='x', label='Predicted Data')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.legend()
plt.show()