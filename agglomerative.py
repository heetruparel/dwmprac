from scipy.spatial.distance import pdist, squareform
import numpy as np

def agglomerative_clustering(X, n_clusters):
    clusters = [[i] for i in range(len(X))]
    distance_matrix = squareform(pdist(X))
    np.fill_diagonal(distance_matrix, np.inf)
    
    while len(clusters) > n_clusters:
        # Find two closest clusters
        i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
        
        # Merge cluster j into cluster i
        clusters[i] = clusters[i] + clusters[j]
        clusters.pop(j)
        
        # Update distances
        for k in range(len(clusters)):
            if k != i:
                dist = np.min([np.linalg.norm(X[p1] - X[p2]) for p1 in clusters[i] for p2 in clusters[k]])
                distance_matrix[i, k] = dist
                distance_matrix[k, i] = dist
        
        # Remove row and column for cluster j
        distance_matrix = np.delete(distance_matrix, j, axis=0)
        distance_matrix = np.delete(distance_matrix, j, axis=1)
        np.fill_diagonal(distance_matrix, np.inf)
    
    # Assign labels
    labels = np.zeros(len(X))
    for cluster_idx, cluster in enumerate(clusters):
        for sample_idx in cluster:
            labels[sample_idx] = cluster_idx
    
    return labels

# Example
X, _ = make_blobs(n_samples=50, centers=3, random_state=42)
labels = agglomerative_clustering(X, n_clusters=3)

# Plot
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.title('Manual Agglomerative Clustering')
plt.show()
