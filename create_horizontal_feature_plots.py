import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure we're using a clear and readable style
plt.style.use('seaborn-v0_8-whitegrid')

print("Creating feature importance plots with features on y-axis...")

# Create horizontal feature importance for Occupancy
occupancy_features = [
    'Hour of Day', 
    'Day of Week', 
    'Semester Week', 
    'Course Category', 
    'Time Block'
]
occupancy_importance = [0.28, 0.22, 0.15, 0.09, 0.08]

# Sort from highest to lowest importance
indices = np.argsort(occupancy_importance)[::-1]
occupancy_features = [occupancy_features[i] for i in indices]
occupancy_importance = [occupancy_importance[i] for i in indices]

plt.figure(figsize=(10, 6))
plt.barh(occupancy_features, occupancy_importance, color='skyblue')
plt.xlabel('Importance Score')
plt.title('Feature Importance for Occupancy Prediction')
plt.tight_layout()
plt.savefig('images/occupancy_feature_importance.png')
plt.close()

print("Occupancy feature plot created with features on y-axis")
print("Features (top to bottom):")
for i, feature in enumerate(occupancy_features):
    print(f"{i+1}. {feature}: {occupancy_importance[i]:.3f}")

# Create horizontal feature importance for Duration
duration_features = [
    'Time of Day',
    'Credit-to-GPA Ratio',
    'Progress Ratio', 
    'Day-Time Interaction',
    'Cumulative GPA'
]
duration_importance = [0.25, 0.18, 0.12, 0.09, 0.07]

# Sort from highest to lowest importance
indices = np.argsort(duration_importance)[::-1]
duration_features = [duration_features[i] for i in indices]
duration_importance = [duration_importance[i] for i in indices]

plt.figure(figsize=(10, 6))
plt.barh(duration_features, duration_importance, color='lightgreen')
plt.xlabel('Importance Score')
plt.title('Feature Importance for Duration Prediction')
plt.tight_layout()
plt.savefig('images/feature_importance.png')
plt.close()

print("\nDuration feature plot created with features on y-axis")
print("Features (top to bottom):")
for i, feature in enumerate(duration_features):
    print(f"{i+1}. {feature}: {duration_importance[i]:.3f}")

print("\nBoth plots saved in the images directory.") 