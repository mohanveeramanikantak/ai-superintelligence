# Text Classification using NLP + Machine Learning

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset
texts = [
    "Win money now", 
    "Limited offer just for you", 
    "Call your friend today", 
    "Meeting scheduled at 10am",
    "Congratulations you won a prize",
    "Let's have lunch tomorrow"
]

labels = [1, 1, 0, 0, 1, 0]  # 1 = Spam, 0 = Not Spam

# Convert text to numerical data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test custom input
test_message = ["You have won a lottery"]
test_vector = vectorizer.transform(test_message)
prediction = model.predict(test_vector)

print("Spam" if prediction[0] == 1 else "Not Spam")
