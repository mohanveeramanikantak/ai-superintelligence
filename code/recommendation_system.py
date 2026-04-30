# Simple Recommendation System using Cosine Similarity

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# User-item matrix (rows = users, cols = items)
# 1 = liked, 0 = not liked
data = np.array([
    [1, 1, 0, 0, 1],  # User 1
    [1, 0, 0, 1, 1],  # User 2
    [0, 1, 1, 0, 0],  # User 3
    [1, 1, 0, 1, 1]   # User 4
])

# Compute similarity between users
similarity = cosine_similarity(data)

print("User Similarity Matrix:\n", similarity)

# Recommend items for user 0
user_index = 0
similar_users = similarity[user_index]

# Get most similar user (excluding itself)
most_similar_user = similar_users.argsort()[-2]

print("Most similar user:", most_similar_user)

# Recommend items liked by similar user but not by current user
recommendations = []

for i in range(len(data[0])):
    if data[user_index][i] == 0 and data[most_similar_user][i] == 1:
        recommendations.append(f"Item {i}")

print("Recommended items:", recommendations)
