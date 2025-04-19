import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler

def find_kmeanspp_centers(X, n_clusters, random_state=42):
    """
    Implement K-means++ initialization method.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The data to cluster
    n_clusters : int
        The number of clusters to form
    random_state : int, optional
        Random state for reproducibility
        
    Returns:
    --------
    centers : array, shape (n_clusters, n_features)
        Initial cluster centers
    """
    np.random.seed(random_state)
    n_samples = X.shape[0]
    
    # Choose first center randomly
    centers = [X[np.random.randint(n_samples)]]
    
    # Choose remaining centers
    for _ in range(1, n_clusters):
        # Calculate distances to nearest center for each point
        distances = pairwise_distances_argmin_min(X, centers)[1]
        
        # Choose next center with probability proportional to distance squared
        probs = distances ** 2 / np.sum(distances ** 2)
        next_center_idx = np.random.choice(n_samples, p=probs)
        centers.append(X[next_center_idx])
    
    return np.array(centers)

def perform_clustering(X, n_clusters=11, init='random', random_state=42):
    """
    Perform K-means clustering and return results.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The data to cluster
    n_clusters : int
        The number of clusters to form
    init : str or array-like
        Initialization method
    random_state : int, optional
        Random state for reproducibility
        
    Returns:
    --------
    kmeans : KMeans object
        Fitted KMeans model
    """
    if init == 'kmeans++':
        init_centers = find_kmeanspp_centers(X, n_clusters, random_state)
        kmeans = KMeans(n_clusters=n_clusters, init=init_centers, n_init=1, random_state=random_state)
    else:
        kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=10, random_state=random_state)
    
    kmeans.fit(X)
    return kmeans

def analyze_clusters(X, kmeans):
    """
    Analyze and print information about the clusters.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The data points
    kmeans : KMeans object
        Fitted KMeans model
    """
    print("\nCluster Analysis:")
    print("----------------")
    
    # Calculate cluster sizes
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    
    # Print cluster information
    for cluster in range(kmeans.n_clusters):
        print(f"\nCluster {cluster + 1}:")
        print(f"Size: {cluster_sizes[cluster]} points")
        print(f"Center: {kmeans.cluster_centers_[cluster]}")
        
        # Calculate average distance to center for points in this cluster
        cluster_points = X[kmeans.labels_ == cluster]
        if len(cluster_points) > 0:
            distances = np.linalg.norm(cluster_points - kmeans.cluster_centers_[cluster], axis=1)
            avg_distance = np.mean(distances)
            print(f"Average distance to center: {avg_distance:.4f}")

def plot_clusters(X, kmeans, title):
    """
    Plot the data points colored by cluster labels with centroids highlighted.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The data points
    kmeans : KMeans object
        Fitted KMeans model
    title : str
        Title for the plot
    """
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='tab20', alpha=0.6)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                c='red', marker='x', s=200, linewidths=3)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter, label='Cluster')
    plt.savefig(f'kmeans_{title.lower().replace(" ", "_")}.png')
    plt.close()

def main():
    # Load the dataset
    data = pd.read_csv('kmeans_dataset.csv')
    X = data.values
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform standard K-means clustering
    kmeans_standard = perform_clustering(X_scaled, init='random')
    print("\nStandard K-means Results:")
    print(f"Total within-cluster sum of squares: {kmeans_standard.inertia_:.2f}")
    print(f"Number of iterations: {kmeans_standard.n_iter_}")
    analyze_clusters(X_scaled, kmeans_standard)
    plot_clusters(X_scaled, kmeans_standard, "Standard K-means Clustering")
    
    # Perform K-means++ clustering
    kmeans_pp = perform_clustering(X_scaled, init='kmeans++')
    print("\nK-means++ Results:")
    print(f"Total within-cluster sum of squares: {kmeans_pp.inertia_:.2f}")
    print(f"Number of iterations: {kmeans_pp.n_iter_}")
    analyze_clusters(X_scaled, kmeans_pp)
    plot_clusters(X_scaled, kmeans_pp, "K-means++ Clustering")

if __name__ == "__main__":
    main() 