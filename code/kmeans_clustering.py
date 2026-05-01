# K-Means Clustering Example

from sklearn.cluster import KMeans
import numpy as np

# Sample data (2D points)
X = np.array([
    [1, 2], [1.5, 1.8], [5, 8],
    [8, 8], [1, 0.6], [9, 11]
])

# Create model
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Results
print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Labels:", kmeans.labels_)

# Predict new point
new_point = [[3, 3]]
print("Predicted cluster:", kmeans.predict(new_point))
