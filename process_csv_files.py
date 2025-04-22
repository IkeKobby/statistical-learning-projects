import pandas as pd
import os

# Create submission directory if it doesn't exist
os.makedirs('submission', exist_ok=True)

# Process occupancy file - round to integers and remove Student_IDs
occupancy_df = pd.read_csv('occupancy_prediction/predictions/predicted_occupancy.csv')
occupancy_df['Occupancy'] = occupancy_df['Occupancy'].round().astype(int)
occupancy_df[['Occupancy']].to_csv('submission/predicted_occupancy.csv', index=False)

# Process duration file - round to 2 decimal places and remove Student_IDs
duration_df = pd.read_csv('duration_prediction/predictions/predicted_duration_in_minutes.csv')
duration_df['Duration_In_Min'] = duration_df['Duration_In_Min'].round(2)
duration_df[['Duration_In_Min']].to_csv('submission/predicted_duration_in_minutes.csv', index=False)

print("Files processed and saved to submission folder:")
print("- submission/predicted_occupancy.csv")
print("- submission/predicted_duration_in_minutes.csv") 