# Linear Regression Example

from sklearn.linear_model import LinearRegression
import numpy as np

# Data (experience → salary)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([30000, 40000, 50000, 60000, 70000])

# Model
model = LinearRegression()
model.fit(X, y)

# Predict
experience = [[6]]
prediction = model.predict(experience)

print(f"Predicted salary for 6 years experience: {prediction[0]}")
