# Sentiment Analysis using TF-IDF + Logistic Regression

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset
texts = [
    "I love this product",
    "This is amazing",
    "Worst experience ever",
    "I hate it",
    "Very good and useful",
    "Not worth the money"
]

labels = [1, 1, 0, 0, 1, 0]  # 1=Positive, 0=Negative

# Convert text to vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3)

# Train
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test new input
test = ["This product is awesome"]
print("Prediction:", model.predict(vectorizer.transform(test)))
