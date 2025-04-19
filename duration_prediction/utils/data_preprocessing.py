import pandas as pd
import numpy as np
import joblib
import sys
import os
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import time

def create_features(df):
    """Create features from the input dataframe"""
    df = df.copy()
    
    # Extract Graduation_Semester and Graduation_Year from Expected_Graduation
    if 'Expected_Graduation' in df.columns:
        df['Expected_Graduation'] = df['Expected_Graduation'].fillna('Unknown')
        df['Graduation_Semester'] = df['Expected_Graduation'].apply(
            lambda x: x.split()[0] if x != 'Unknown' and len(x.split()) > 1 else 'Unknown'
        )
        df['Graduation_Year'] = df['Expected_Graduation'].apply(
            lambda x: x.split()[1] if x != 'Unknown' and len(x.split()) > 1 else np.nan
        )
        # Convert graduation year to numeric, coerce errors to NaN
        df['Graduation_Year'] = pd.to_numeric(df['Graduation_Year'], errors='coerce')
    
    # Time-related features
    if 'Check_In_Time' in df.columns:
        # Convert time to minutes from midnight
        df['Check_In_Time'] = df['Check_In_Time'].fillna('00:00')
        df['time_of_day_minutes'] = df['Check_In_Time'].apply(
            lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]) if ':' in str(x) else 0
        )
        
        # Create time categories (morning, afternoon, evening)
        df['time_category'] = pd.cut(
            df['time_of_day_minutes'],
            bins=[0, 6*60, 12*60, 18*60, 24*60],
            labels=['night', 'morning', 'afternoon', 'evening']
        )
        
        # Create a day/time interaction if day of week is available
        if 'Check_In_Date' in df.columns:
            df['check_in_day'] = pd.to_datetime(df['Check_In_Date'], errors='coerce')
            df['day_of_week'] = df['check_in_day'].dt.dayofweek
            df['day_time_interaction'] = df['day_of_week'].astype(str) + "_" + df['time_category'].astype(str)
    
    # Create years to graduation
    if 'Graduation_Year' in df.columns:
        # Assuming current_year is 2016 (based on your dataset)
        current_year = 2016
        df['years_to_graduation'] = df['Graduation_Year'] - current_year
        df['years_to_graduation'] = df['years_to_graduation'].clip(lower=0)
    
    # Academic performance features
    if 'Term_GPA' in df.columns and 'Term_Credit_Hours' in df.columns:
        df['credit_to_gpa_ratio'] = df['Term_Credit_Hours'] / (df['Term_GPA'] + 0.1)
    
    if 'Cumulative_GPA' in df.columns and 'Term_GPA' in df.columns:
        df['cumulative_to_term_gpa_ratio'] = df['Cumulative_GPA'] / (df['Term_GPA'] + 0.1)
    
    # Engagement indicator (assuming higher credit hours indicate higher engagement)
    if 'Term_Credit_Hours' in df.columns:
        df['high_engagement'] = (df['Term_Credit_Hours'] > df['Term_Credit_Hours'].median()).astype(int)
    
    # Course complexity indicator
    if 'Course_Code_by_Thousands' in df.columns:
        df['course_level'] = df['Course_Code_by_Thousands'].fillna(0).astype(int)
        df['advanced_course'] = (df['course_level'] >= 3000).astype(int)
    
    # Create cyclical features for day of week if available
    if 'day_of_week' in df.columns:
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Time of day as cyclical feature
    if 'time_of_day_minutes' in df.columns:
        df['time_sin'] = np.sin(2 * np.pi * df['time_of_day_minutes'] / (24 * 60))
        df['time_cos'] = np.cos(2 * np.pi * df['time_of_day_minutes'] / (24 * 60))
        
        # Add peak hour indicators (common study times)
        df['peak_hour'] = ((df['time_of_day_minutes'] >= 10*60) & 
                          (df['time_of_day_minutes'] <= 16*60)).astype(int)
        
        df['evening_peak'] = ((df['time_of_day_minutes'] >= 18*60) & 
                             (df['time_of_day_minutes'] <= 22*60)).astype(int)
    
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
    
    # Apply feature engineering
    print("Applying feature engineering...")
    start_time = time.time()
    test_features = create_features(test_data)
    print(f"Feature engineering completed in {time.time() - start_time:.2f} seconds")
    
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
            'Student_IDs': test_data['Student_IDs'],
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
    parser = argparse.ArgumentParser(description='Predict Duration_In_Min using trained model')
    parser.add_argument('--model', type=str, help='Path to trained model file', default='best_duration_predictor.pkl')
    parser.add_argument('--data', type=str, help='Path to test data file', default='LC_test.csv')
    parser.add_argument('--output', type=str, help='Path to save predictions', default='predictions.csv')
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