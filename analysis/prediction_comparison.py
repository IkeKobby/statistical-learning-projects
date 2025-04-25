import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Read the prediction files
prev_duration = pd.read_csv('submission/previous_duration_in_minutes_predictions.csv')
curr_duration = pd.read_csv('submission/predicted_duration_in_minutes.csv')
prev_occupancy = pd.read_csv('submission/previous_predicted_occupancy.csv')
curr_occupancy = pd.read_csv('submission/predicted_occupancy.csv')

# Standardize column names
prev_duration.columns = ['Duration_In_Min']
if 'Duration_In_Minutes' in prev_duration.columns:
    prev_duration.rename(columns={'Duration_In_Minutes': 'Duration_In_Min'}, inplace=True)

# Combine data for comparison
duration_comparison = pd.DataFrame({
    'Previous': prev_duration['Duration_In_Min'],
    'Current': curr_duration['Duration_In_Min'],
})

occupancy_comparison = pd.DataFrame({
    'Previous': prev_occupancy['Occupancy'],
    'Current': curr_occupancy['Occupancy'],
})

# Calculate differences
duration_comparison['Difference'] = duration_comparison['Current'] - duration_comparison['Previous']
occupancy_comparison['Difference'] = occupancy_comparison['Current'] - occupancy_comparison['Previous']

# 1. Overall Statistics
print("==== Duration Prediction Statistics ====")
print("\nPrevious Duration Predictions:")
print(duration_comparison['Previous'].describe())
print("\nCurrent Duration Predictions:")
print(duration_comparison['Current'].describe())
print("\nDifference (Current - Previous):")
print(duration_comparison['Difference'].describe())

print("\n\n==== Occupancy Prediction Statistics ====")
print("\nPrevious Occupancy Predictions:")
print(occupancy_comparison['Previous'].describe())
print("\nCurrent Occupancy Predictions:")
print(occupancy_comparison['Current'].describe())
print("\nDifference (Current - Previous):")
print(occupancy_comparison['Difference'].describe())

# 2. Distribution Plots
plt.figure(figsize=(15, 10))

# Duration distributions
plt.subplot(2, 2, 1)
sns.histplot(duration_comparison['Previous'], kde=True, color='blue', alpha=0.5, label='Previous')
sns.histplot(duration_comparison['Current'], kde=True, color='red', alpha=0.5, label='Current')
plt.title('Duration Prediction Distribution')
plt.xlabel('Duration (minutes)')
plt.ylabel('Count')
plt.legend()

# Duration difference
plt.subplot(2, 2, 2)
sns.histplot(duration_comparison['Difference'], kde=True, color='purple')
plt.axvline(x=0, color='black', linestyle='--')
plt.title('Duration Prediction Differences')
plt.xlabel('Difference (Current - Previous)')
plt.ylabel('Count')

# Occupancy distributions
plt.subplot(2, 2, 3)
sns.histplot(occupancy_comparison['Previous'], kde=True, bins=30, color='blue', alpha=0.5, label='Previous')
sns.histplot(occupancy_comparison['Current'], kde=True, bins=30, color='red', alpha=0.5, label='Current')
plt.title('Occupancy Prediction Distribution')
plt.xlabel('Occupancy (people)')
plt.ylabel('Count')
plt.legend()

# Occupancy difference
plt.subplot(2, 2, 4)
sns.histplot(occupancy_comparison['Difference'], kde=True, bins=30, color='purple')
plt.axvline(x=0, color='black', linestyle='--')
plt.title('Occupancy Prediction Differences')
plt.xlabel('Difference (Current - Previous)')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('analysis/distribution_comparison.png')

# 3. Scatter plots to compare predictions
plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
plt.scatter(duration_comparison['Previous'], duration_comparison['Current'], alpha=0.5)
plt.plot([duration_comparison['Previous'].min(), duration_comparison['Previous'].max()], 
         [duration_comparison['Previous'].min(), duration_comparison['Previous'].max()], 
         'r--')
plt.title('Duration: Previous vs Current Predictions')
plt.xlabel('Previous Prediction (minutes)')
plt.ylabel('Current Prediction (minutes)')

plt.subplot(1, 2, 2)
plt.scatter(occupancy_comparison['Previous'], occupancy_comparison['Current'], alpha=0.5)
plt.plot([occupancy_comparison['Previous'].min(), occupancy_comparison['Previous'].max()], 
         [occupancy_comparison['Previous'].min(), occupancy_comparison['Previous'].max()], 
         'r--')
plt.title('Occupancy: Previous vs Current Predictions')
plt.xlabel('Previous Prediction (people)')
plt.ylabel('Current Prediction (people)')

plt.tight_layout()
plt.savefig('analysis/scatter_comparison.png')

# 4. Calculate statistical tests to compare distributions
# Kolmogorov-Smirnov test for duration
ks_duration = stats.ks_2samp(duration_comparison['Previous'], duration_comparison['Current'])
print("\nKolmogorov-Smirnov Test for Duration Distributions:")
print(f"Statistic: {ks_duration.statistic:.4f}, p-value: {ks_duration.pvalue:.4e}")
print("Interpretation: " + ("Distributions are different" if ks_duration.pvalue < 0.05 else "Distributions are similar"))

# Kolmogorov-Smirnov test for occupancy
ks_occupancy = stats.ks_2samp(occupancy_comparison['Previous'], occupancy_comparison['Current'])
print("\nKolmogorov-Smirnov Test for Occupancy Distributions:")
print(f"Statistic: {ks_occupancy.statistic:.4f}, p-value: {ks_occupancy.pvalue:.4e}")
print("Interpretation: " + ("Distributions are different" if ks_occupancy.pvalue < 0.05 else "Distributions are similar"))

# 5. Additional analysis - range, variance, and extreme predictions
print("\n\n==== Additional Analysis ====")
print("\nDuration Range Comparison:")
print(f"Previous: {duration_comparison['Previous'].min():.2f} to {duration_comparison['Previous'].max():.2f}")
print(f"Current: {duration_comparison['Current'].min():.2f} to {duration_comparison['Current'].max():.2f}")
print(f"Variance Previous: {duration_comparison['Previous'].var():.2f}")
print(f"Variance Current: {duration_comparison['Current'].var():.2f}")

print("\nOccupancy Range Comparison:")
print(f"Previous: {occupancy_comparison['Previous'].min():.2f} to {occupancy_comparison['Previous'].max():.2f}")
print(f"Current: {occupancy_comparison['Current'].min():.2f} to {occupancy_comparison['Current'].max():.2f}")
print(f"Variance Previous: {occupancy_comparison['Previous'].var():.2f}")
print(f"Variance Current: {occupancy_comparison['Current'].var():.2f}")

# 6. Correlation between predictions
dur_corr = duration_comparison['Previous'].corr(duration_comparison['Current'])
occ_corr = occupancy_comparison['Previous'].corr(occupancy_comparison['Current'])

print("\nCorrelation between previous and current predictions:")
print(f"Duration correlation: {dur_corr:.4f}")
print(f"Occupancy correlation: {occ_corr:.4f}")

# 7. Percentage of predictions that changed significantly
dur_significant_change = (abs(duration_comparison['Difference']) > 15).mean() * 100
occ_significant_change = (abs(occupancy_comparison['Difference']) > 3).mean() * 100

print(f"\nPercentage of duration predictions with significant change (>15 min): {dur_significant_change:.2f}%")
print(f"Percentage of occupancy predictions with significant change (>3 people): {occ_significant_change:.2f}%")

# 8. Box plots for more distribution insight
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
boxplot_data_duration = [duration_comparison['Previous'], duration_comparison['Current']]
plt.boxplot(boxplot_data_duration, labels=['Previous', 'Current'])
plt.title('Duration Prediction Distribution (Box Plot)')
plt.ylabel('Duration (minutes)')

plt.subplot(1, 2, 2)
boxplot_data_occupancy = [occupancy_comparison['Previous'], occupancy_comparison['Current']]
plt.boxplot(boxplot_data_occupancy, labels=['Previous', 'Current'])
plt.title('Occupancy Prediction Distribution (Box Plot)')
plt.ylabel('Occupancy (people)')

plt.tight_layout()
plt.savefig('analysis/boxplot_comparison.png')

# Print overall summary
print("\n==== Overall Summary ====")
print("Duration predictions:")
if abs(duration_comparison['Difference'].mean()) > 5:
    print(f"- Mean prediction changed by {abs(duration_comparison['Difference'].mean()):.2f} minutes")
    if duration_comparison['Difference'].mean() > 0:
        print("- Current model predicts LONGER durations on average")
    else:
        print("- Current model predicts SHORTER durations on average")
else:
    print("- Mean prediction remained relatively similar")

if duration_comparison['Current'].var() > duration_comparison['Previous'].var():
    print("- Current model has MORE variance in predictions")
else:
    print("- Current model has LESS variance in predictions")

print("\nOccupancy predictions:")
if abs(occupancy_comparison['Difference'].mean()) > 1:
    print(f"- Mean prediction changed by {abs(occupancy_comparison['Difference'].mean()):.2f} people")
    if occupancy_comparison['Difference'].mean() > 0:
        print("- Current model predicts HIGHER occupancy on average")
    else:
        print("- Current model predicts LOWER occupancy on average")
else:
    print("- Mean prediction remained relatively similar")

if occupancy_comparison['Current'].var() > occupancy_comparison['Previous'].var():
    print("- Current model has MORE variance in predictions")
else:
    print("- Current model has LESS variance in predictions")

# Save results to text file
with open('analysis/prediction_analysis_summary.txt', 'w') as f:
    f.write("# Prediction Model Comparison Summary\n\n")
    f.write("## Duration Prediction Analysis\n")
    f.write(f"Previous mean: {duration_comparison['Previous'].mean():.2f} minutes\n")
    f.write(f"Current mean: {duration_comparison['Current'].mean():.2f} minutes\n")
    f.write(f"Mean difference: {duration_comparison['Difference'].mean():.2f} minutes\n")
    f.write(f"Previous standard deviation: {duration_comparison['Previous'].std():.2f}\n")
    f.write(f"Current standard deviation: {duration_comparison['Current'].std():.2f}\n")
    f.write(f"Correlation between models: {dur_corr:.4f}\n")
    f.write(f"Percentage of significantly different predictions: {dur_significant_change:.2f}%\n\n")
    
    f.write("## Occupancy Prediction Analysis\n")
    f.write(f"Previous mean: {occupancy_comparison['Previous'].mean():.2f} people\n")
    f.write(f"Current mean: {occupancy_comparison['Current'].mean():.2f} people\n")
    f.write(f"Mean difference: {occupancy_comparison['Difference'].mean():.2f} people\n")
    f.write(f"Previous standard deviation: {occupancy_comparison['Previous'].std():.2f}\n")
    f.write(f"Current standard deviation: {occupancy_comparison['Current'].std():.2f}\n")
    f.write(f"Correlation between models: {occ_corr:.4f}\n")
    f.write(f"Percentage of significantly different predictions: {occ_significant_change:.2f}%\n") 