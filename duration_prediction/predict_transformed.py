import pandas as pd
import numpy as np
import joblib
import sys
import os
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import time

def create_minimal_features(df):
    """Create minimal features for the already transformed data"""
    df = df.copy()
    
    # Check if the Student_IDs column exists
    if 'Student_IDs' not in df.columns:
        # Generate sequence IDs if not present
        df['Student_IDs'] = np.arange(len(df))
        print("Created sequential Student_IDs as they were not found in the input data")
    
    # Extract semester and year from Expected_Graduation
    if 'Expected_Graduation' in df.columns:
        df['Graduation_Semester'] = df['Expected_Graduation'].str.split(' ').str[0]
        df['Graduation_Year'] = df['Expected_Graduation'].str.split(' ').str[-1].astype(int)
    
    # Create time-related features
    if 'hour' in df.columns and 'minute' in df.columns:
        df['time_of_day_minutes'] = df['hour'] * 60 + df['minute']
        
        # Create a feature for time proximity to noon (peak hours)
        df['time_to_noon'] = abs(12 - df['hour'] - (df['minute'] / 60))
        
        # Create a feature for whether the time is during peak hours (10am-2pm)
        df['is_peak_hours'] = ((df['hour'] >= 10) & (df['hour'] <= 14)).astype(int)
    
    # Create interactions between time and day features
    if 'Day_Of_Week' in df.columns and 'time_of_day_minutes' in df.columns:
        df['day_time_interaction'] = df['Day_Of_Week'] * df['time_of_day_minutes']
    
    # Calculate the difference between current and expected graduation
    if 'Graduation_Year' in df.columns:
        # Extract year from the semester (e.g., "Fall 2017" -> 2017)
        if 'Semester' in df.columns:
            current_year = df['Semester'].str.split(' ').str[-1].astype(int).iloc[0]
        else:
            # Use a reasonable default if not present
            current_year = 2017  # Based on data
        df['years_to_graduation'] = df['Graduation_Year'] - current_year
    
    # Academic performance features
    if 'Term_GPA' in df.columns and 'Term_Credit_Hours' in df.columns:
        df['credit_to_gpa_ratio'] = df['Term_Credit_Hours'] / (df['Term_GPA'] + 0.1)  # Adding 0.1 to avoid division by zero
    
    if 'Cumulative_GPA' in df.columns and 'Term_GPA' in df.columns:
        df['cumulative_to_term_gpa_ratio'] = df['Cumulative_GPA'] / (df['Term_GPA'] + 0.1)
    
    # Time in school indicator (proxy for experience)
    if 'Total_Credit_Hours_Earned' in df.columns and 'Term_Credit_Hours' in df.columns:
        df['progress_ratio'] = df['Total_Credit_Hours_Earned'] / (df['Term_Credit_Hours'] + 1)
    
    # Create week of semester percentage
    if 'Semester_Week' in df.columns:
        df['semester_progress'] = df['Semester_Week'] / 17  # Assuming a 17-week semester
    
    # Student engagement indicator (speculative)
    if 'Term_Credit_Hours' in df.columns and 'Term_GPA' in df.columns:
        df['student_engagement'] = df['Term_Credit_Hours'] * df['Term_GPA']
    
    # Create categorical mappings for ordinal variables
    if 'Class_Standing' in df.columns:
        class_standing_map = {
            'Freshman': 1, 
            'Sophomore': 2, 
            'Junior': 3, 
            'Senior': 4, 
            'Graduate': 5
        }
        df['class_standing_numeric'] = df['Class_Standing'].map(class_standing_map)
    
    # Create cyclical features for days of week (to capture cyclical nature)
    if 'Day_Of_Week' in df.columns:
        df['day_sin'] = np.sin(2 * np.pi * df['Day_Of_Week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['Day_Of_Week'] / 7)
    
    # Create a feature for whether the course is STEM or not
    if 'course_category' in df.columns:
        df['is_stem'] = df['course_category'].isin(['Science', 'Technology', 'Engineering', 'Mathematics']).astype(int)
    
    # Create a feature for course complexity based on course level
    if 'course_level' in df.columns:
        course_level_map = {
            'Freshman level Course': 1,
            'Sophomore level Course': 2,
            'Junior level Course': 3,
            'Senior level Course': 4,
            'Unknown course number': 2.5  # Assigning a middle value
        }
        df['course_complexity'] = df['course_level'].map(course_level_map)
    
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
    """Load test data, apply minimal feature engineering, and make predictions"""
    print(f"Loading transformed test data from {test_file}...")
    test_data = pd.read_csv(test_file)
    print(f"Test data shape: {test_data.shape}")
    
    # Apply minimal feature engineering
    print("Applying minimal feature engineering for transformed data...")
    start_time = time.time()
    test_features = create_minimal_features(test_data)
    print(f"Feature processing completed in {time.time() - start_time:.2f} seconds")
    
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
            pred = model.predict(test_features)
            all_preds.append(pred)
        
        # Combine predictions using weights
        predictions = np.zeros(len(test_features))
        for i, pred in enumerate(all_preds):
            predictions += weights[i] * pred
    else:
        predictions = model.predict(test_features)
    
    # Add predictions to the test data
    test_data['Duration_In_Min_Predicted'] = predictions
    
    # If target variable is present, evaluate the model
    if 'Duration_In_Min' in test_data.columns:
        rmse = np.sqrt(mean_squared_error(test_data['Duration_In_Min'], predictions))
        mae = mean_absolute_error(test_data['Duration_In_Min'], predictions)
        r2 = r2_score(test_data['Duration_In_Min'], predictions)
        
        print(f"\nModel Evaluation on Test Set:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R^2: {r2:.4f}")
    
    # Save predictions to output file if provided
    if output_file:
        # For submission format, extract only required columns
        submission_df = pd.DataFrame({
            'Student_IDs': test_data['Student_IDs'] if 'Student_IDs' in test_data.columns else np.arange(len(test_data)),
            'Duration_In_Min': predictions.clip(0)  # Ensure no negative predictions
        })
        submission_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
        # Also save full predictions with all columns for analysis
        full_output_file = output_file.replace('.csv', '_full.csv')
        test_data.to_csv(full_output_file, index=False)
        print(f"Full predictions with all features saved to {full_output_file}")
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description='Predict Duration_In_Min using trained model on transformed data')
    parser.add_argument('--model', type=str, help='Path to trained model file', default='best_duration_predictor.pkl')
    parser.add_argument('--data', type=str, help='Path to transformed test data file', default='lc_transformed_test_data.csv')
    parser.add_argument('--output', type=str, help='Path to save predictions', default='predictions_transformed.csv')
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Model file {args.model} not found.")
        
        # Try to find available model files
        model_files = [f for f in os.listdir() if f.endswith('.pkl')]
        
        if model_files:
            print(f"Available model files: {model_files}")
            args.model = model_files[0]
            print(f"Using {args.model} instead.")
        else:
            print("No model files found. Please train a model first.")
            sys.exit(1)
    
    # Special handling for weighted ensemble
    if "weighted_ensemble" in args.model:
        print("Using weighted ensemble model...")
        
        # Load ensemble configuration
        try:
            ensemble_config = joblib.load('weighted_ensemble_config.pkl')
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