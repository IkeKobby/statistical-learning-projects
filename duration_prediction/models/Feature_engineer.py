import pandas as pd
import numpy as np

print("Loading and preparing data...")

# Load the data
data = pd.read_csv('../processed_data/duration_in_minutes_train_data.csv')

# Function to create features (same as in model_building.py)
def create_features(df):
    df_new = df.copy()
    
    # Extract semester and year from Expected_Graduation
    df_new['Graduation_Semester'] = df_new['Expected_Graduation'].str.split(' ').str[0]
    df_new['Graduation_Year'] = df_new['Expected_Graduation'].str.split(' ').str[-1].astype(int)
    
    # Create time-related features
    df_new['time_of_day_minutes'] = df_new['hour'] * 60 + df_new['minute']
    
    # Create interactions between time and day features
    df_new['day_time_interaction'] = df_new['Day_Of_Week'] * df_new['time_of_day_minutes']
    
    # Calculate the difference between current and expected graduation
    current_year = 2016  # Assuming all records are from 2016 based on the 'Semester' column
    df_new['years_to_graduation'] = df_new['Graduation_Year'] - current_year
    
    # Academic performance features
    df_new['credit_to_gpa_ratio'] = df_new['Term_Credit_Hours'] / (df_new['Term_GPA'] + 0.1)  # Adding 0.1 to avoid division by zero
    df_new['cumulative_to_term_gpa_ratio'] = df_new['Cumulative_GPA'] / (df_new['Term_GPA'] + 0.1)
    
    # Time in school indicator (proxy for experience)
    df_new['progress_ratio'] = df_new['Total_Credit_Hours_Earned'] / (df_new['Term_Credit_Hours'] + 1)
    
    # Create week of semester percentage
    df_new['semester_progress'] = df_new['Semester_Week'] / 17  # Assuming a 17-week semester
    
    # Student engagement indicator (speculative)
    df_new['student_engagement'] = df_new['Term_Credit_Hours'] * df_new['Term_GPA']
    
    # Create categorical mappings for ordinal variables
    class_standing_map = {
        'Freshman': 1, 
        'Sophomore': 2, 
        'Junior': 3, 
        'Senior': 4, 
        'Graduate': 5,
        'Other': 2.5
    }
    df_new['class_standing_numeric'] = df_new['Class_Standing'].map(class_standing_map)
    
    # Create cyclical features for days of week (to capture cyclical nature)
    df_new['day_sin'] = np.sin(2 * np.pi * df_new['Day_Of_Week'] / 7)
    df_new['day_cos'] = np.cos(2 * np.pi * df_new['Day_Of_Week'] / 7)
    
    # Create a feature for time proximity to noon (peak hours)
    df_new['time_to_noon'] = abs(12 - df_new['hour'] - (df_new['minute'] / 60))
    
    # Create a feature for whether the time is during peak hours (10am-2pm)
    df_new['is_peak_hours'] = ((df_new['hour'] >= 10) & (df_new['hour'] <= 14)).astype(int)
    
    # Create a feature for whether the course is STEM or not
    df_new['is_stem'] = df_new['course_category'].isin(['Science', 'Technology', 'Engineering', 'Mathematics']).astype(int)
    
    # Create a feature for course complexity based on course level
    course_level_map = {
        'Freshman level Course': 1,
        'Sophomore level Course': 2,
        'Junior level Course': 3,
        'Senior level Course': 4,
        'Graduate level Course': 5,
        'Unknown course number': 2.5  # Assigning a middle value
    }
    df_new['course_complexity'] = df_new['course_level'].map(course_level_map)
    
    return df_new

# Create features
data_engineered = create_features(data)
data_engineered.to_csv('../processed_data/engineered_data.csv', index=False)