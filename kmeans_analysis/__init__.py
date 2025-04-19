"""
K-means Clustering Analysis Package

This package provides tools for analyzing Learning Center data using K-means clustering.
"""

from .models.kmeans_analysis import perform_clustering, analyze_clusters, plot_clusters
from .utils.data_preprocessing import preprocess_clustering_data

__version__ = '0.1.0'
__all__ = [
    'perform_clustering',
    'analyze_clusters',
    'plot_clusters',
    'preprocess_clustering_data'
] 