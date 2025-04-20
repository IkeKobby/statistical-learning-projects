import pandas as pd
import numpy as np
import joblib
import os
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_features(df):
    """Create features for the model"""
    df = df.copy()
    
    # Extract semester and year from Expected_Graduation
    df['Graduation_Semester'] = df['Expected_Graduation'].str.split(' ').str[0]
    df['Graduation_Year'] = df['Expected_Graduation'].str.split(' ').str[-1].astype(int)
    
    # Create time-related features
    df['time_of_day_minutes'] = df['hour'] * 60 + df['minute']
    
    # Create a feature for time proximity to noon (peak hours)
    df['time_to_noon'] = abs(12 - df['hour'] - (df['minute'] / 60))
    
    # Create a feature for whether the time is during peak hours (10am-2pm)
    df['is_peak_hours'] = ((df['hour'] >= 10) & (df['hour'] <= 14)).astype(int)
    
    # Create interactions between time and day features
    df['day_time_interaction'] = df['Day_Of_Week'] * df['time_of_day_minutes']
    
    # Calculate the difference between current and expected graduation
    current_year = df['Semester'].str.split(' ').str[-1].astype(int).iloc[0]
    df['years_to_graduation'] = df['Graduation_Year'] - current_year
    
    # Academic performance features
    df['credit_to_gpa_ratio'] = df['Term_Credit_Hours'] / (df['Term_GPA'] + 0.1)  # Adding 0.1 to avoid division by zero
    df['cumulative_to_term_gpa_ratio'] = df['Cumulative_GPA'] / (df['Term_GPA'] + 0.1)
    
    # Time in school indicator (proxy for experience)
    df['progress_ratio'] = df['Total_Credit_Hours_Earned'] / (df['Term_Credit_Hours'] + 1)
    
    # Create week of semester percentage
    df['semester_progress'] = df['Semester_Week'] / 17  # Assuming a 17-week semester
    
    # Student engagement indicator (speculative)
    df['student_engagement'] = df['Term_Credit_Hours'] * df['Term_GPA']
    
    # Create categorical mappings for ordinal variables
    class_standing_map = {
        'Freshman': 1, 
        'Sophomore': 2, 
        'Junior': 3, 
        'Senior': 4, 
        'Graduate': 5,
        'Other': 3  # Default value for any other category
    }
    df['class_standing_numeric'] = df['Class_Standing'].map(class_standing_map)
    
    # Create cyclical features for days of week (to capture cyclical nature)
    df['day_sin'] = np.sin(2 * np.pi * df['Day_Of_Week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['Day_Of_Week'] / 7)
    
    # Create a feature for whether the course is STEM or not
    df['is_stem'] = df['course_category'].isin(['Science', 'Technology', 'Engineering', 'Mathematics']).astype(int)
    
    # Create a feature for course complexity based on course level
    course_level_map = {
        'Freshman level Course': 1,
        'Sophomore level Course': 2,
        'Junior level Course': 3,
        'Senior level Course': 4,
        'Graduate level Course': 5,
        'Unknown course number': 2.5  # Assigning a middle value
    }
    df['course_complexity'] = df['course_level'].map(course_level_map)
    
    # Hour-of-day patterns might be especially important for occupancy
    # Let's create one-hot encoding for hour blocks
    hour_blocks = pd.cut(df['hour'], bins=[6, 10, 14, 18, 22], labels=['morning', 'midday', 'afternoon', 'evening'])
    hour_dummies = pd.get_dummies(hour_blocks, prefix='hour_block')
    df = pd.concat([df, hour_dummies], axis=1)
    
    # Day of week might show weekly patterns in occupancy
    day_dummies = pd.get_dummies(df['Day_Of_Week'], prefix='day')
    df = pd.concat([df, day_dummies], axis=1)
    
    return df

def make_predictions(model_path, data_path, output_dir):
    """Make predictions using a trained model on new data"""
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    
    # Process the data
    logger.info("Processing features...")
    data_processed = create_features(data)
    
    # Define features
    categorical_cols = ['Semester', 'Class_Standing', 'Expected_Graduation', 'Gender', 
                       'Graduation_Semester', 'course_category', 'time_bin', 'course_level']
    drop_cols = categorical_cols + ['minute']
    
    # Keep only numeric columns and engineered features
    numeric_cols = data_processed.select_dtypes(include=['float64', 'int64']).columns
    
    # Check if 'occupancy' is in the data (for validation) or not (for prediction)
    is_validation = 'occupancy' in data_processed.columns
    
    if is_validation:
        logger.info("Target column 'occupancy' found in data, will calculate metrics")
        X = data_processed[numeric_cols].drop(columns=['occupancy'])
        y_true = data_processed['occupancy']
    else:
        logger.info("No target column found, performing prediction only")
        X = data_processed[numeric_cols]
    
    # Add any one-hot encoded columns we created
    one_hot_cols = [col for col in data_processed.columns if col.startswith('hour_block_') or col.startswith('day_')]
    for col in one_hot_cols:
        if col in data_processed.columns:
            X[col] = data_processed[col]
    
    # Load imputer
    imputer_path = os.path.join(os.path.dirname(model_path), 'imputer_occupancy.pkl')
    if os.path.exists(imputer_path):
        logger.info(f"Loading imputer from {imputer_path}")
        imputer = joblib.load(imputer_path)
        X_imputed = pd.DataFrame(imputer.transform(X), columns=X.columns)
    else:
        logger.warning(f"Imputer not found at {imputer_path}, using simple imputation")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = model.predict(X_imputed)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions
    # If we have the original IDs, use them, otherwise create sequential IDs
    if 'Student_IDs' in data.columns:
        ids = data['Student_IDs']
    else:
        logger.warning("No 'Student_IDs' column found, creating sequential IDs")
        ids = pd.Series(range(1, len(predictions) + 1))
    
    predictions_df = pd.DataFrame({
        'Student_IDs': ids,
        'Occupancy': predictions
    })
    
    predictions_path = os.path.join(output_dir, 'predicted_occupancy.csv')
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to {predictions_path}")
    
    # If we have true values, calculate metrics
    if is_validation:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)
        
        logger.info("Evaluation metrics:")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"R²: {r2:.4f}")
        
        # Create a DataFrame with actual and predicted values
        validation_df = pd.DataFrame({
            'Actual': y_true,
            'Predicted': predictions
        })
        
        if 'Student_IDs' in data.columns:
            validation_df['Student_IDs'] = data['Student_IDs']
        
        validation_path = os.path.join(output_dir, 'validation_results.csv')
        validation_df.to_csv(validation_path, index=False)
        logger.info(f"Validation results saved to {validation_path}")
    
    return predictions_df

def main():
    parser = argparse.ArgumentParser(description='Make predictions using a trained occupancy model')
    parser.add_argument('--model', type=str, default='occupancy_prediction/predictions/best_occupancy_predictor.pkl', 
                        help='Path to the trained model')
    parser.add_argument('--data', type=str, default='occupancy_prediction/processed_data/occupancy_test_data.csv',
                        help='Path to the test/validation data')
    parser.add_argument('--output_dir', type=str, default='occupancy_prediction/predictions',
                        help='Directory to save the predictions')
    args = parser.parse_args()
    
    make_predictions(args.model, args.data, args.output_dir)

if __name__ == '__main__':
    main() 