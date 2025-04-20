import pandas as pd
import numpy as np
import joblib
import os
import sys
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import argparse

# Add parent directory to python path to import from sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_preprocessing import create_features

def load_data(data_path):
    """Load the dataset"""
    try:
        print(f"Loading data from {data_path}...")
        data = pd.read_csv(data_path)
        print(f"Data loaded successfully with shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def load_model(model_path):
    """Load the trained model"""
    try:
        print(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
        print(f"Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def perform_cross_validation(model, data, n_folds=10, model_name="Best Model", processed_data=None):
    """Perform k-fold cross-validation on the entire dataset"""
    print(f"\nPerforming {n_folds}-fold cross-validation for {model_name}...")
    
    # Prepare features and target
    if processed_data is not None:
        # Use the pre-processed data (for XGBoost)
        X = processed_data
        y = data['Duration_In_Min']
        print(f"Using pre-processed data with shape: {X.shape}")
    else:
        # Use raw data (for models that handle categorical features internally)
        X = data.drop('Duration_In_Min', axis=1)
        y = data['Duration_In_Min']
    
    # Define cross-validation strategy
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Calculate cross-validation scores
    rmse_scores = -cross_val_score(model, X, y, cv=kf, 
                                  scoring='neg_root_mean_squared_error', 
                                  n_jobs=-1)
    mae_scores = -cross_val_score(model, X, y, cv=kf, 
                                 scoring='neg_mean_absolute_error', 
                                 n_jobs=-1)
    r2_scores = cross_val_score(model, X, y, cv=kf, 
                               scoring='r2', 
                               n_jobs=-1)
    
    # Make predictions for each fold (for plotting)
    y_pred = cross_val_predict(model, X, y, cv=kf, n_jobs=-1)
    
    # Print results
    print(f"\n{model_name} Cross-Validation Results:")
    print(f"RMSE: {rmse_scores.mean():.4f} (±{rmse_scores.std():.4f})")
    print(f"MAE: {mae_scores.mean():.4f} (±{mae_scores.std():.4f})")
    print(f"R²: {r2_scores.mean():.4f} (±{r2_scores.std():.4f})")
    
    return {
        'model_name': model_name,
        'rmse': rmse_scores,
        'mae': mae_scores,
        'r2': r2_scores,
        'predictions': y_pred,
        'actual': y
    }

def train_new_xgboost_model(data):
    """Train a fresh XGBoost model with optimized parameters"""
    print("\nTraining a new XGBoost model for comparison...")
    
    # Prepare features and target
    X = data.drop('Duration_In_Min', axis=1)
    y = data['Duration_In_Min']
    
    # Handle categorical features (XGBoost needs numerical features)
    categorical_cols = X.select_dtypes(include=['object']).columns
    print(f"Preprocessing categorical features: {list(categorical_cols)}")
    
    # Use pandas get_dummies to one-hot encode categorical features
    X_processed = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    print(f"Data shape after one-hot encoding: {X_processed.shape}")
    
    # Set up XGBoost model with optimized parameters
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=2,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=1.0,
        random_state=42
    )
    
    print("XGBoost model configured with optimized parameters")
    return model, X_processed

def visualize_cv_results(cv_results, output_dir='duration_prediction/visualizations', prefix='cv'):
    """Generate visualizations for cross-validation results"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set the style for plots
    sns.set(style="whitegrid")
    
    # Predicted vs Actual
    plt.figure(figsize=(10, 8))
    plt.scatter(cv_results['actual'], cv_results['predictions'], alpha=0.5)
    plt.plot([cv_results['actual'].min(), cv_results['actual'].max()], 
             [cv_results['actual'].min(), cv_results['actual'].max()], 'r--')
    plt.xlabel('Actual Duration (minutes)')
    plt.ylabel('Predicted Duration (minutes)')
    plt.title(f'{cv_results["model_name"]}: Actual vs Predicted Duration')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_{cv_results["model_name"].lower().replace(" ", "_")}_predictions_vs_actual.png'))
    plt.close()
    
    # Residuals
    residuals = cv_results['actual'] - cv_results['predictions']
    plt.figure(figsize=(10, 6))
    plt.scatter(cv_results['predictions'], residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Duration (minutes)')
    plt.ylabel('Residuals (minutes)')
    plt.title(f'{cv_results["model_name"]}: Residual Plot')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_{cv_results["model_name"].lower().replace(" ", "_")}_residuals.png'))
    plt.close()
    
    # Distribution of residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals (minutes)')
    plt.ylabel('Frequency')
    plt.title(f'{cv_results["model_name"]}: Distribution of Residuals')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_{cv_results["model_name"].lower().replace(" ", "_")}_residuals_distribution.png'))
    plt.close()
    
    # Distribution of predictions vs actual
    plt.figure(figsize=(10, 6))
    sns.kdeplot(cv_results['actual'], label='Actual')
    sns.kdeplot(cv_results['predictions'], label='Predicted')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Density')
    plt.title(f'{cv_results["model_name"]}: Distribution of Actual vs Predicted Duration')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_{cv_results["model_name"].lower().replace(" ", "_")}_distribution_comparison.png'))
    plt.close()
    
    print(f"Visualizations for {cv_results['model_name']} saved to {output_dir}")

def compare_models(model1_results, model2_results, output_dir='duration_prediction/visualizations'):
    """Compare the performance of two models"""
    print("\nComparing model performance...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create comparison table
    model_comparison = pd.DataFrame({
        model1_results['model_name']: {
            'RMSE': model1_results['rmse'].mean(),
            'RMSE Std': model1_results['rmse'].std(),
            'MAE': model1_results['mae'].mean(),
            'MAE Std': model1_results['mae'].std(),
            'R²': model1_results['r2'].mean(),
            'R² Std': model1_results['r2'].std()
        },
        model2_results['model_name']: {
            'RMSE': model2_results['rmse'].mean(),
            'RMSE Std': model2_results['rmse'].std(),
            'MAE': model2_results['mae'].mean(),
            'MAE Std': model2_results['mae'].std(),
            'R²': model2_results['r2'].mean(),
            'R² Std': model2_results['r2'].std()
        }
    }).T
    
    print("\nModel Comparison:")
    print(model_comparison)
    
    # Save comparison table
    model_comparison.to_csv(os.path.join(output_dir, 'model_comparison_cv.csv'))
    
    # Create bar charts for RMSE and R²
    plt.figure(figsize=(12, 6))
    
    # RMSE comparison
    plt.subplot(1, 2, 1)
    models = [model1_results['model_name'], model2_results['model_name']]
    rmse_values = [model1_results['rmse'].mean(), model2_results['rmse'].mean()]
    rmse_std = [model1_results['rmse'].std(), model2_results['rmse'].std()]
    
    bars = plt.bar(models, rmse_values, yerr=rmse_std, capsize=10, color=['skyblue', 'lightgreen'])
    plt.ylabel('RMSE (lower is better)')
    plt.title('RMSE Comparison')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.2f}', ha='center', va='bottom')
    
    # R² comparison
    plt.subplot(1, 2, 2)
    r2_values = [model1_results['r2'].mean(), model2_results['r2'].mean()]
    r2_std = [model1_results['r2'].std(), model2_results['r2'].std()]
    
    bars = plt.bar(models, r2_values, yerr=r2_std, capsize=10, color=['skyblue', 'lightgreen'])
    plt.ylabel('R² (higher is better)')
    plt.title('R² Comparison')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_metrics.png'))
    plt.close()
    
    # Compare residual distributions
    plt.figure(figsize=(12, 6))
    residuals1 = model1_results['actual'] - model1_results['predictions']
    residuals2 = model2_results['actual'] - model2_results['predictions']
    
    sns.kdeplot(residuals1, label=model1_results['model_name'])
    sns.kdeplot(residuals2, label=model2_results['model_name'])
    plt.xlabel('Residuals (minutes)')
    plt.ylabel('Density')
    plt.title('Comparison of Residual Distributions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_residuals.png'))
    plt.close()
    
    print(f"Model comparison visualizations saved to {output_dir}")
    
    return model_comparison

def make_predictions(model, test_file, output_file):
    """Make predictions on test data and save results"""
    print(f"\nMaking predictions on {test_file}...")
    
    # Load test data
    test_data = pd.read_csv(test_file)
    print(f"Test data loaded with shape: {test_data.shape}")
    
    # Import the create_minimal_features function from predict_transformed.py
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from predict_transformed import create_minimal_features
    
    # Create features for prediction
    print("Applying feature engineering to test data...")
    test_features = create_minimal_features(test_data)
    
    # Make predictions
    print("Generating predictions...")
    predictions = model.predict(test_features)
    
    # Create predictions file without IDs
    pred_df = pd.DataFrame({
        'predicted_duration_in_minutes': predictions.clip(0)  # Ensure no negative predictions
    })
    
    # Save predictions
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pred_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    # Also save with IDs for submission
    submission_df = pd.DataFrame({
        'Student_IDs': test_data['Student_IDs'] if 'Student_IDs' in test_data.columns else range(len(test_data)),
        'Duration_In_Min': predictions.clip(0)
    })
    
    submission_file = os.path.join(os.path.dirname(output_file), 'submission.csv')
    submission_df.to_csv(submission_file, index=False)
    print(f"Submission file saved to {submission_file}")
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description='Perform cross-validation and make predictions with best model')
    parser.add_argument('--model', type=str, default='duration_prediction/models/saved_models/best_duration_predictor.pkl',
                        help='Path to the trained best model')
    parser.add_argument('--data', type=str, default='duration_prediction/processed_data/engineered_data.csv',
                        help='Path to the engineered training data')
    parser.add_argument('--test', type=str, default='duration_prediction/lc_transformed_test_data.csv',
                        help='Path to the test data for prediction')
    parser.add_argument('--output', type=str, default='duration_prediction/predictions/predicted_duration_in_minutes.csv',
                        help='Path to save the predictions')
    parser.add_argument('--folds', type=int, default=10,
                        help='Number of cross-validation folds')
    parser.add_argument('--skip-predictions', action='store_true',
                        help='Skip making predictions on test data')
    parser.add_argument('--compare', action='store_true', default=True,
                        help='Compare with freshly trained XGBoost model')
    
    args = parser.parse_args()
    
    # Load the data
    data = load_data(args.data)
    
    # Load the best model
    best_model = load_model(args.model)
    
    # Perform cross-validation on the best model
    cv_results_best = perform_cross_validation(best_model, data, args.folds, "LightGBM (Best Model)")
    
    # Visualize cross-validation results for best model
    visualize_cv_results(cv_results_best)
    
    # Compare with XGBoost if requested
    if args.compare:
        # Train a new XGBoost model
        xgb_model, X_processed = train_new_xgboost_model(data)
        
        # Perform cross-validation on XGBoost model
        cv_results_xgb = perform_cross_validation(xgb_model, data, args.folds, "XGBoost", X_processed)
        
        # Visualize cross-validation results for XGBoost model
        visualize_cv_results(cv_results_xgb, prefix='xgb')
        
        # Compare the two models
        model_comparison = compare_models(cv_results_best, cv_results_xgb)
    
    # Make predictions on test data if file exists
    try:
        if os.path.exists(args.test) and not args.skip_predictions:
            predictions = make_predictions(best_model, args.test, args.output)
        else:
            print(f"\nSkipping predictions: Test file '{args.test}' not found or predictions skipped.")
    except Exception as e:
        print(f"\nError during prediction: {e}")
        print("Continuing without making predictions...")
    
    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main() 