import numpy as np
import matplotlib.pyplot as plt

def initialize_centroids(X, k):
    np.random.seed(42)
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

def closest_centroid(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def compute_centroids(X, labels, k):
    centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return centroids

def kmeans_manual(X, k, n_iters=100):
    centroids = initialize_centroids(X, k)
    
    for _ in range(n_iters):
        labels = closest_centroid(X, centroids)
        new_centroids = compute_centroids(X, labels, k)
        
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return labels, centroids

# Example
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
labels, centroids = kmeans_manual(X, k=4)

# Plot
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200)
plt.title('Manual K-Means Clustering')
plt.show()
