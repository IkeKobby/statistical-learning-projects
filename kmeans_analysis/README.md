# K-means Clustering Analysis

This project performs K-means clustering analysis on the provided dataset using both standard K-means and K-means++ initialization methods.

## Files
- `kmeans_analysis.py`: Main script for performing K-means clustering
- `kmeans_dataset.csv`: Input dataset for clustering
- `requirements.txt`: Python package dependencies

## Setup
1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the analysis script:
```bash
python kmeans_analysis.py
```

The script will:
1. Load the dataset from `kmeans_dataset.csv`
2. Perform standard K-means clustering
3. Perform K-means++ clustering
4. Generate visualization plots for both methods
5. Print clustering metrics including within-cluster sum of squares and number of iterations

## Output
The script generates two visualization files:
- `kmeans_standard_k-means_clustering.png`: Results from standard K-means
- `kmeans_k-means++_clustering.png`: Results from K-means++

## Analysis
The script compares the performance of standard K-means and K-means++ initialization methods by:
- Calculating the total within-cluster sum of squares
- Counting the number of iterations required for convergence
- Visualizing the cluster assignments and centroids 