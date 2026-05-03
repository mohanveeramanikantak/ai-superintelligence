# Decision Tree Classifier Example

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset (age, salary → buy or not)
X = [
    [25, 50000], [30, 60000], [45, 80000],
    [35, 120000], [22, 20000], [40, 70000]
]
y = [0, 1, 1, 1, 0, 1]  # 0 = No, 1 = Yes

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Test
print("Prediction for [28, 65000]:", model.predict([[28, 65000]]))
