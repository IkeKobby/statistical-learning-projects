import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_predictions(predictions_file, output_dir):
    """Analyze occupancy predictions and create visualizations"""
    logger.info(f"Loading predictions from {predictions_file}")
    df = pd.read_csv(predictions_file)
    
    # Create visualizations directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set styling for plots
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Basic statistics
    logger.info("\nBasic statistics of predictions:")
    occupancy_stats = df['Occupancy'].describe()
    logger.info(f"Minimum: {occupancy_stats['min']:.2f} students")
    logger.info(f"Maximum: {occupancy_stats['max']:.2f} students")
    logger.info(f"Mean: {occupancy_stats['mean']:.2f} students")
    logger.info(f"Median: {occupancy_stats['50%']:.2f} students")
    logger.info(f"Standard deviation: {occupancy_stats['std']:.2f} students")
    logger.info(f"Total predictions: {len(df)} records")
    
    # 1. Distribution of occupancy predictions (histogram)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Occupancy'], kde=True, bins=30)
    plt.title('Distribution of Predicted Occupancy', fontsize=16)
    plt.xlabel('Occupancy (Number of Students)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    hist_path = os.path.join(output_dir, 'occupancy_predictions_histogram.png')
    plt.savefig(hist_path)
    plt.close()
    logger.info(f"Histogram saved to {hist_path}")
    
    # 2. Occupancy ranges with percentages
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50']
    df['occupancy_bin'] = pd.cut(df['Occupancy'], bins=bins, labels=labels)
    
    bin_counts = df['occupancy_bin'].value_counts().sort_index()
    bin_percentages = (bin_counts / bin_counts.sum() * 100).round(2)
    
    plt.figure(figsize=(12, 6))
    ax = bin_percentages.plot(kind='bar', color='skyblue')
    plt.title('Distribution of Predicted Occupancy by Range', fontsize=16)
    plt.xlabel('Occupancy Range (Number of Students)', fontsize=14)
    plt.ylabel('Percentage of Time Slots', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add percentage labels on top of each bar
    for i, v in enumerate(bin_percentages):
        if not np.isnan(v):  # Only add labels to non-NaN values
            ax.text(i, v + 0.5, f'{v}%', ha='center')
    
    plt.tight_layout()
    ranges_path = os.path.join(output_dir, 'occupancy_predictions_by_range.png')
    plt.savefig(ranges_path)
    plt.close()
    logger.info(f"Range distribution chart saved to {ranges_path}")
    
    # 3. Cumulative distribution of occupancy
    plt.figure(figsize=(10, 6))
    sorted_occupancy = np.sort(df['Occupancy'])
    cumulative = np.arange(1, len(sorted_occupancy) + 1) / len(sorted_occupancy)
    
    plt.plot(sorted_occupancy, cumulative, linewidth=2, color='darkblue')
    plt.grid(True, alpha=0.3)
    plt.title('Cumulative Distribution of Predicted Occupancy', fontsize=16)
    plt.xlabel('Occupancy (Number of Students)', fontsize=14)
    plt.ylabel('Cumulative Probability', fontsize=14)
    
    # Add vertical lines at key percentiles
    q25 = np.percentile(df['Occupancy'], 25)
    q50 = np.percentile(df['Occupancy'], 50)
    q75 = np.percentile(df['Occupancy'], 75)
    q90 = np.percentile(df['Occupancy'], 90)
    
    plt.axvline(x=q25, color='r', linestyle='--', label=f'25th Percentile: {q25:.1f}')
    plt.axvline(x=q50, color='g', linestyle='--', label=f'Median: {q50:.1f}')
    plt.axvline(x=q75, color='orange', linestyle='--', label=f'75th Percentile: {q75:.1f}')
    plt.axvline(x=q90, color='purple', linestyle='--', label=f'90th Percentile: {q90:.1f}')
    plt.legend()
    
    plt.tight_layout()
    cdf_path = os.path.join(output_dir, 'occupancy_cumulative_distribution.png')
    plt.savefig(cdf_path)
    plt.close()
    logger.info(f"Cumulative distribution chart saved to {cdf_path}")
    
    # 4. Box plot of predictions
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df['Occupancy'])
    plt.title('Box Plot of Predicted Occupancy', fontsize=16)
    plt.ylabel('Occupancy (Number of Students)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text for key statistics
    plt.text(0.02, 0.95, f"Mean: {occupancy_stats['mean']:.2f}", transform=plt.gca().transAxes)
    plt.text(0.02, 0.90, f"Median: {occupancy_stats['50%']:.2f}", transform=plt.gca().transAxes)
    plt.text(0.02, 0.85, f"Std Dev: {occupancy_stats['std']:.2f}", transform=plt.gca().transAxes)
    
    plt.tight_layout()
    boxplot_path = os.path.join(output_dir, 'occupancy_boxplot.png')
    plt.savefig(boxplot_path)
    plt.close()
    logger.info(f"Box plot saved to {boxplot_path}")
    
    # Print occupancy ranges
    logger.info("\nPercentage of time slots in each occupancy range:")
    for label, percentage in zip(bin_percentages.index, bin_percentages):
        if not np.isnan(percentage):
            logger.info(f"{label} students: {percentage:.2f}%")
    
    # Save summary statistics to CSV
    summary_df = pd.DataFrame({
        'Statistic': ['Min', 'Max', 'Mean', 'Median', 'Standard Deviation', 'Total Predictions'],
        'Value': [
            f"{occupancy_stats['min']:.2f} students",
            f"{occupancy_stats['max']:.2f} students",
            f"{occupancy_stats['mean']:.2f} students",
            f"{occupancy_stats['50%']:.2f} students",
            f"{occupancy_stats['std']:.2f} students",
            f"{len(df)} records"
        ]
    })
    
    summary_path = os.path.join(output_dir, 'occupancy_prediction_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary statistics saved to {summary_path}")
    
    # Save bin percentages to CSV
    bin_percentages_df = pd.DataFrame({
        'Occupancy Range': bin_percentages.index,
        'Percentage': bin_percentages.values
    }).dropna()
    
    bins_path = os.path.join(output_dir, 'occupancy_ranges_distribution.csv')
    bin_percentages_df.to_csv(bins_path, index=False)
    logger.info(f"Occupancy range distribution saved to {bins_path}")
    
    logger.info("\nAnalysis completed! Visualizations and reports saved to the output directory.")

def main():
    parser = argparse.ArgumentParser(description='Analyze occupancy predictions')
    parser.add_argument('--predictions', type=str, default='occupancy_prediction/predictions/predicted_occupancy.csv',
                        help='Path to the predictions file')
    parser.add_argument('--output_dir', type=str, default='occupancy_prediction/visualizations',
                        help='Directory to save visualizations and reports')
    args = parser.parse_args()
    
    analyze_predictions(args.predictions, args.output_dir)

if __name__ == '__main__':
    main() 