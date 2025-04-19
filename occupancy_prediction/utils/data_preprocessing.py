import pandas as pd
import numpy as np
import joblib
import sys
import os
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

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
    # Extract year from the semester (e.g., "Fall 2017" -> 2017)
    if 'Semester' in df.columns:
        current_year = df['Semester'].str.split(' ').str[-1].astype(int).iloc[0]
    else:
        # Use a reasonable default if not present
        current_year = 2017  # Based on observed data
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

def load_model(model_path):
    """Load the trained model from a pickle file"""
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def make_predictions(model, test_file, output_file=None, weighted_ensemble=False, ensemble_models=None, weights=None):
    """Load test data, apply feature engineering, and make predictions"""
    print(f"Loading test data from {test_file}...")
    test_data = pd.read_csv(test_file)
    print(f"Test data shape: {test_data.shape}")
    
    # Store Student_IDs if present, otherwise create them
    if 'Student_IDs' not in test_data.columns:
        # Generate sequence IDs if not present
        test_data['Student_IDs'] = np.arange(len(test_data))
        print("Created sequential Student_IDs as they were not found in the input data")
    
    # Keep a copy of the IDs
    student_ids = test_data['Student_IDs'].copy() if 'Student_IDs' in test_data.columns else np.arange(len(test_data))
    
    # Apply feature engineering
    print("Applying feature engineering...")
    start_time = time.time()
    test_features = create_features(test_data)
    print(f"Feature processing completed in {time.time() - start_time:.2f} seconds")
    
    # Drop columns not needed for modeling
    categorical_cols = ['Semester', 'Class_Standing', 'Expected_Graduation', 'Gender', 
                       'Graduation_Semester', 'course_category', 'time_bin', 'course_level']
    drop_cols = categorical_cols + ['minute']  # Drop minute as we have time_of_day_minutes
    
    # Remove Student_IDs if present to avoid feature mismatch with the model
    if 'Student_IDs' in test_features.columns:
        test_features = test_features.drop(columns=['Student_IDs'])
    
    # Keep only numeric columns and engineered features
    numeric_cols = test_features.select_dtypes(include=['float64', 'int64']).columns
    test_features_filtered = test_features[numeric_cols]
    
    # Add any one-hot encoded columns we created
    one_hot_cols = [col for col in test_features.columns if col.startswith('hour_block_') or col.startswith('day_')]
    for col in one_hot_cols:
        if col in test_features.columns:
            test_features_filtered[col] = test_features[col]
    
    # Check for and handle missing values
    missing_values = test_features_filtered.isnull().sum().sum()
    if missing_values > 0:
        print(f"Found {missing_values} missing values. Applying imputation...")
        try:
            # Load the imputer
            imputer = joblib.load('models/imputer_occupancy.pkl')
            
            # Get the feature names used during training
            imputer_feature_names = imputer.feature_names_in_
            
            # Ensure test features have the same columns in the same order
            missing_cols = set(imputer_feature_names) - set(test_features_filtered.columns)
            for col in missing_cols:
                test_features_filtered[col] = 0  # Add missing columns with default values
            
            # Select and order columns to match the training data
            test_features_filtered = test_features_filtered[imputer_feature_names]
            
            # Apply imputation
            test_features_filtered = pd.DataFrame(
                imputer.transform(test_features_filtered), 
                columns=imputer_feature_names
            )
            print("Imputation completed.")
        except FileNotFoundError:
            print("Imputer not found. Using simple median imputation.")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            test_features_filtered = pd.DataFrame(
                imputer.fit_transform(test_features_filtered),
                columns=test_features_filtered.columns
            )
    
    # Make predictions
    print("Making predictions...")
    
    if weighted_ensemble:
        if not ensemble_models or not weights:
            print("Error: Weighted ensemble requires models and weights")
            sys.exit(1)
        
        # Load each model and make predictions
        all_preds = []
        for model_path in ensemble_models:
            model = load_model(model_path)
            pred = model.predict(test_features_filtered)
            all_preds.append(pred)
        
        # Combine predictions using weights
        predictions = np.zeros(len(test_features_filtered))
        for i, pred in enumerate(all_preds):
            predictions += weights[i] * pred
    else:
        predictions = model.predict(test_features_filtered)
    
    # Round to nearest integer (occupancy must be a whole number)
    predictions = np.round(predictions).astype(int)
    
    # Ensure no negative predictions
    predictions = np.maximum(predictions, 1)  # Minimum occupancy is 1
    
    # Add predictions to the test data
    test_data['Occupancy_Predicted'] = predictions
    
    # If target variable is present, evaluate the model
    if 'occupancy' in test_data.columns:
        rmse = np.sqrt(mean_squared_error(test_data['occupancy'], predictions))
        mae = mean_absolute_error(test_data['occupancy'], predictions)
        r2 = r2_score(test_data['occupancy'], predictions)
        
        print(f"\nModel Evaluation on Test Set:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R^2: {r2:.4f}")
    
    # Save predictions to output file if provided
    if output_file:
        # For submission format, extract only required columns
        submission_df = pd.DataFrame({
            'Student_IDs': student_ids,
            'Occupancy': predictions
        })
        submission_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
        # Also save full predictions with all columns for analysis
        full_output_file = output_file.replace('.csv', '_full.csv')
        test_data.to_csv(full_output_file, index=False)
        print(f"Full predictions with all features saved to {full_output_file}")
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description='Predict occupancy using trained model')
    parser.add_argument('--model', type=str, help='Path to trained model file', default='models/best_occupancy_predictor.pkl')
    parser.add_argument('--data', type=str, help='Path to test data file', default='lc_transformed_test_data.csv')
    parser.add_argument('--output', type=str, help='Path to save predictions', default='occupancy_predictions.csv')
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Model file {args.model} not found.")
        
        # Try to find available model files
        model_files = [f for f in os.listdir('models') if f.endswith('.pkl') and 'occupancy' in f]
        
        if model_files:
            print(f"Available model files: {model_files}")
            args.model = os.path.join('models', model_files[0])
            print(f"Using {args.model} instead.")
        else:
            print("No occupancy model files found. Please train a model first.")
            sys.exit(1)
    
    # Special handling for weighted ensemble
    if "weighted_ensemble" in args.model:
        print("Using weighted ensemble model...")
        
        # Load ensemble configuration
        try:
            ensemble_config = joblib.load('weighted_ensemble_config_occupancy.pkl')
            ensemble_models = ensemble_config['models']
            weights = ensemble_config['weights']
            
            print(f"Ensemble models: {ensemble_models}")
            print(f"Weights: {weights}")
            
            # Use None as the primary model for the ensemble
            make_predictions(None, args.data, args.output, 
                            weighted_ensemble=True,
                            ensemble_models=ensemble_models, 
                            weights=weights)
        except Exception as e:
            print(f"Error loading ensemble configuration: {e}")
            sys.exit(1)
    else:
        # Load the model and make predictions
        model = load_model(args.model)
        make_predictions(model, args.data, args.output)

if __name__ == "__main__":
    main() 