import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set a nice style for the plot
plt.style.use('seaborn-v0_8-whitegrid')

# Load the feature importance data
importance_df = pd.read_csv('occupancy_prediction/visualizations/occupancy_feature_importance.csv')

# Sort by importance (highest first)
importance_df = importance_df.sort_values('Importance', ascending=False)

# Get the top 15 features for better readability
top_n = 15
top_features = importance_df.head(top_n)

# Reverse the order for plotting - this ensures highest is at the top
# We need to reverse the data since barh plots from bottom to top
reversed_features = top_features.iloc[::-1].reset_index(drop=True)

# Create the figure
plt.figure(figsize=(12, 10))

# Create horizontal bar plot with blue color
plt.barh(reversed_features['Feature'], reversed_features['Importance'], color='blue', edgecolor='darkblue')

# Customize the plot
plt.xlabel('Importance Score', fontsize=12)
plt.title('Top 15 Features for Occupancy Prediction', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Add values at the end of each bar
for i, v in enumerate(reversed_features['Importance']):
    plt.text(v + 0.003, i, f'{v:.3f}', va='center', color='black', fontweight='bold')

# Ensure the y-axis labels are readable
plt.yticks(fontsize=11)

# Adjust layout and limits for better appearance
plt.xlim(0, top_features['Importance'].max() * 1.1)
plt.tight_layout()

# Save the plot
plt.savefig('images/occupancy_feature_importance.png', dpi=300, bbox_inches='tight')

print("Feature importance plot created with features on y-axis (highest importance at top) and saved to images/occupancy_feature_importance.png")

# Display feature names and importance values
print("\nTop 15 features by importance (highest first):")
for i, (feature, importance) in enumerate(zip(top_features['Feature'], top_features['Importance'])):
    print(f"{i+1}. {feature}: {importance:.4f}") 