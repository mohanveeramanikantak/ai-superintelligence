# AI Superintelligence – Advanced ML Example
# Classification using Logistic Regression

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset (hours studied vs pass/fail)
X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [0, 0, 0, 0, 1, 1, 1, 1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Predict new value
hours = 5
prediction = model.predict([[hours]])

if prediction[0] == 1:
    print(f"Student studying {hours} hours is likely to PASS")
else:
    print(f"Student studying {hours} hours is likely to FAIL")
