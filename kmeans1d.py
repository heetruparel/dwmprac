import pandas as pd
import matplotlib.pyplot as plt
import random

# Step 1: Read Excel file
df = pd.read_excel(r'C:\Users\Cinepix\Downloads\KMeans1D.xlsx')  # Replace with actual filename
data = df.iloc[:, 0].tolist()  # Get first column as list

# Step 2: Set number of clusters
k = int(input("Enter number of clusters (k): "))  # User specifies number of clusters

# Initialize means by randomly selecting k points
means = random.sample(data, k)

def assign_clusters(data, means):
    clusters = [[] for _ in range(len(means))]
    for point in data:
        # Find closest mean
        distances = [abs(point - m) for m in means]
        closest_mean_idx = distances.index(min(distances))
        clusters[closest_mean_idx].append(point)
    return clusters

def compute_mean(cluster):
    return sum(cluster) / len(cluster) if cluster else 0

# Step 3: Iterate until means stabilize
for i in range(10):  # Limit to 10 iterations max
    clusters = assign_clusters(data, means)
    new_means = [compute_mean(cluster) for cluster in clusters]
    
    # Check for convergence
    if new_means == means:
        break  # Converged
    
    means = new_means

# Step 4: Print results
print("Final clusters:")
for i, cluster in enumerate(clusters, 1):
    print(f"Cluster {i}:", sorted(cluster))
for i, mean in enumerate(means, 1):
    print(f"Mean {i}:", mean)

# Step 5: Visualize
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta']  # Add more colors if needed
for i, cluster in enumerate(clusters):
    plt.scatter(cluster, [0]*len(cluster), color=colors[i % len(colors)], label=f'Cluster {i+1}')
plt.scatter(means, [0]*len(means), color='black', label='Centers', marker='x', s=100)

plt.legend()
plt.title(f"1D K-Means Clustering (k={k})")
plt.yticks([])  # Hide Y-axis since it's 1D
plt.xlabel("Values")
plt.show()
