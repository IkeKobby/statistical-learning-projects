import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create visualizations directory if it doesn't exist
os.makedirs('duration_prediction/visualizations', exist_ok=True)

# Load the predictions
print("Loading predictions from duration_prediction/predictions/predicted_duration_in_minutes.csv")
df = pd.read_csv('duration_prediction/predictions/predicted_duration_in_minutes.csv')

# Basic statistics
print("\nBasic statistics of predictions:")
print(f"Minimum: {df['Duration_In_Min'].min():.2f} minutes")
print(f"Maximum: {df['Duration_In_Min'].max():.2f} minutes")
print(f"Mean: {df['Duration_In_Min'].mean():.2f} minutes")
print(f"Median: {df['Duration_In_Min'].median():.2f} minutes")
print(f"Standard deviation: {df['Duration_In_Min'].std():.2f} minutes")
print(f"Total predictions: {len(df)} records")

# Create a histogram of predictions
print("\nCreating histogram of predictions...")
plt.figure(figsize=(10, 6))
sns.histplot(df['Duration_In_Min'], kde=True, bins=30)
plt.title('Distribution of Predicted Session Durations')
plt.xlabel('Duration (minutes)')
plt.ylabel('Frequency')
plt.savefig('duration_prediction/visualizations/predicted_durations_histogram.png')
plt.close()

# Create bins for duration
bins = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270]
labels = ['0-30', '31-60', '61-90', '91-120', '121-150', '151-180', '181-210', '211-240', '241-270']
df['duration_bin'] = pd.cut(df['Duration_In_Min'], bins=bins, labels=labels)

# Calculate percentage in each bin
bin_counts = df['duration_bin'].value_counts().sort_index()
bin_percentages = (bin_counts / bin_counts.sum() * 100).round(2)

# Create a bar plot showing percentage in each bin
print("\nCreating bar plot of duration ranges...")
plt.figure(figsize=(12, 6))
ax = bin_percentages.plot(kind='bar', color='skyblue')
plt.title('Distribution of Predicted Session Durations by Time Range')
plt.xlabel('Duration Range (minutes)')
plt.ylabel('Percentage of Sessions')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add percentage labels on top of each bar
for i, v in enumerate(bin_percentages):
    ax.text(i, v + 0.5, f'{v}%', ha='center')

plt.tight_layout()
plt.savefig('duration_prediction/visualizations/predicted_durations_by_range.png')
plt.close()

print('\nPrediction distribution by time range:')
for label, percentage in zip(labels, bin_percentages):
    print(f'{label} minutes: {percentage}%')

# Create a box plot
print("\nCreating box plot of predictions...")
plt.figure(figsize=(10, 6))
sns.boxplot(y=df['Duration_In_Min'])
plt.title('Box Plot of Predicted Session Durations')
plt.ylabel('Duration (minutes)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('duration_prediction/visualizations/predicted_durations_boxplot.png')
plt.close()

print("\nAnalysis completed! Visualizations saved to duration_prediction/visualizations/") 