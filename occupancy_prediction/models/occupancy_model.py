import pandas as pd
import numpy as np
import joblib
import os
import time
import argparse
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.impute import SimpleImputer

# Set up basic logging to console
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import CatBoost, but continue without it if not available
try:
    from catboost import CatBoostRegressor
    has_catboost = True
except ImportError:
    logger.warning("CatBoost not installed, will skip this model")
    has_catboost = False

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

def visualize_data(df, output_dir="visualizations"):
    """Generate visualizations for exploratory data analysis"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set the style for the plots
    sns.set(style="whitegrid")
    
    # Distribution of occupancy
    plt.figure(figsize=(10, 6))
    sns.histplot(df['occupancy'], kde=True)
    plt.title('Distribution of Occupancy')
    plt.xlabel('Occupancy (Number of Students)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'occupancy_distribution.png'))
    plt.close()
    
    # Occupancy by hour of day
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='hour', y='occupancy', data=df)
    plt.title('Occupancy by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Occupancy')
    plt.savefig(os.path.join(output_dir, 'occupancy_by_hour.png'))
    plt.close()
    
    # Occupancy by day of week
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Day_Of_Week', y='occupancy', data=df)
    plt.title('Occupancy by Day of Week')
    plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
    plt.ylabel('Occupancy')
    plt.savefig(os.path.join(output_dir, 'occupancy_by_day.png'))
    plt.close()
    
    # Occupancy by time bin
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='time_bin', y='occupancy', data=df)
    plt.title('Occupancy by Time of Day')
    plt.xlabel('Time of Day')
    plt.ylabel('Occupancy')
    plt.savefig(os.path.join(output_dir, 'occupancy_by_timebin.png'))
    plt.close()
    
    # Occupancy by course category
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='course_category', y='occupancy', data=df)
    plt.title('Occupancy by Course Category')
    plt.xlabel('Course Category')
    plt.ylabel('Occupancy')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, 'occupancy_by_course.png'))
    plt.close()
    
    # Correlation heatmap
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(14, 12))
    correlation = df[numerical_cols].corr()
    mask = np.triu(correlation)
    sns.heatmap(correlation, annot=False, mask=mask, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix_occupancy.png'))
    plt.close()
    
    # Occupancy trend over semester week
    plt.figure(figsize=(12, 6))
    df_grouped = df.groupby('Semester_Week')['occupancy'].mean().reset_index()
    sns.lineplot(x='Semester_Week', y='occupancy', data=df_grouped, marker='o')
    plt.title('Average Occupancy by Week of Semester')
    plt.xlabel('Week of Semester')
    plt.ylabel('Average Occupancy')
    plt.savefig(os.path.join(output_dir, 'occupancy_by_week.png'))
    plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and return metrics"""
    predictions = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions
    }

def visualize_results(y_test, predictions, output_dir="occupancy_prediction/visualizations"):
    """Generate visualizations of model performance"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Actual vs Predicted plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Occupancy')
    plt.ylabel('Predicted Occupancy')
    plt.title('Actual vs Predicted Occupancy')
    plt.savefig(os.path.join(output_dir, 'occupancy_predictions_vs_actual.png'))
    plt.close()
    
    # Residual plot
    residuals = y_test - predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Occupancy')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.savefig(os.path.join(output_dir, 'occupancy_residuals.png'))
    plt.close()
    
    # Histogram of residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.savefig(os.path.join(output_dir, 'occupancy_residuals_distribution.png'))
    plt.close()
    
    # Distribution of predictions vs actual
    plt.figure(figsize=(10, 6))
    sns.kdeplot(y_test, label='Actual')
    sns.kdeplot(predictions, label='Predicted')
    plt.xlabel('Occupancy')
    plt.ylabel('Density')
    plt.title('Distribution of Actual vs Predicted Occupancy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'occupancy_distribution_comparison.png'))
    plt.close()

def feature_importance_plot(model, feature_names, output_dir="visualizations"):
    """Generate feature importance plot"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        # For models that don't have feature_importances_ attribute
        logger.warning("Model doesn't have feature_importances_ attribute, skipping feature importance plot")
        return
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    
    # Plot the top 20 feature importances
    top_n = min(20, len(feature_names))
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances for Occupancy Prediction')
    plt.barh(range(top_n), importances[indices[:top_n]], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]])
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'occupancy_feature_importance.png'))
    plt.close()
    
    # Also save a CSV with all feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    importance_df.to_csv(os.path.join(output_dir, 'occupancy_feature_importance.csv'), index=False)

def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names):
    """Train and evaluate multiple models and return the best one"""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
    }
    
    # Add CatBoost if available
    if has_catboost:
        models['CatBoost'] = CatBoostRegressor(iterations=100, learning_rate=0.1, random_seed=42, verbose=0)
    
    results = {}
    best_rmse = float('inf')
    best_model_name = None
    best_model = None
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate on test set
        logger.info(f"Evaluating {name}...")
        eval_results = evaluate_model(model, X_test, y_test)
        eval_results['training_time'] = training_time
        results[name] = eval_results
        
        logger.info(f"{name} - RMSE: {eval_results['rmse']:.4f}, MAE: {eval_results['mae']:.4f}, R²: {eval_results['r2']:.4f}, Training Time: {training_time:.2f}s")
        
        # Check if this is the best model so far
        if eval_results['rmse'] < best_rmse:
            best_rmse = eval_results['rmse']
            best_model_name = name
            best_model = model
    
    logger.info(f"Best model: {best_model_name} with RMSE: {best_rmse:.4f}")
    
    # Create a DataFrame with results for easy comparison
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'RMSE': [results[model]['rmse'] for model in results],
        'MAE': [results[model]['mae'] for model in results],
        'R²': [results[model]['r2'] for model in results],
        'Training Time (s)': [results[model]['training_time'] for model in results]
    })
    
    # Save results to CSV
    results_df.to_csv('model_comparison_occupancy.csv', index=False)
    logger.info("Model comparison results saved to model_comparison_occupancy.csv")
    
    # Generate feature importance plot for the best model
    if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']:
        feature_importance_plot(best_model, feature_names)
    
    return best_model, best_model_name, results

def tune_hyperparameters(X_train, y_train, best_model_name):
    """Perform hyperparameter tuning for the best model"""
    logger.info(f"Starting hyperparameter tuning for {best_model_name}...")
    
    if best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = RandomForestRegressor(random_state=42)
        
    elif best_model_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        }
        model = GradientBoostingRegressor(random_state=42)
        
    elif best_model_name == 'XGBoost':
        logger.info("Tuning XGBoost hyperparameters...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        model = xgb.XGBRegressor(random_state=42)
        
    elif best_model_name == 'LightGBM':
        logger.info("Tuning LightGBM hyperparameters...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'num_leaves': [31, 50, 70, 90],
            'min_child_samples': [20, 30, 50],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        model = lgb.LGBMRegressor(random_state=42)
        
    else:
        logger.info(f"Hyperparameter tuning not implemented for {best_model_name}")
        return None
    
    # Use randomized search for efficiency
    from sklearn.model_selection import RandomizedSearchCV
    
    grid_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=20,  # Number of parameter settings sampled
        scoring='neg_root_mean_squared_error',
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best RMSE: {-grid_search.best_score_:.4f}")
    
    # Save best parameters to file
    with open(f'best_params_{best_model_name.lower().replace(" ", "_")}_occupancy.txt', 'w') as f:
        f.write(f"Best parameters for {best_model_name}:\n")
        for param, value in grid_search.best_params_.items():
            f.write(f"{param}: {value}\n")
        f.write(f"\nBest CV RMSE: {-grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def main():
    parser = argparse.ArgumentParser(description='Train a model to predict occupancy in the Learning Center')
    parser.add_argument('--data', type=str, default='occupancy_prediction/processed_data/occupancy_engineered_data.csv', help='Path to training data')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--output_dir', type=str, default='occupancy_prediction/predictions', help='Directory to save predictions and models')
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Occupancy statistics: \n{df['occupancy'].describe()}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    logger.info(f"Missing values in each column:\n{missing_values[missing_values > 0]}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('occupancy_prediction/visualizations', exist_ok=True)
    
    # Visualize data if requested
    if args.visualize:
        logger.info("Generating visualizations...")
        visualize_data(df)
    
    # Create features
    logger.info("Creating features...")
    df_processed = create_features(df)
    
    # Define features and target
    target = 'occupancy'
    
    # Drop columns not needed for modeling
    categorical_cols = ['Semester', 'Class_Standing', 'Expected_Graduation', 'Gender', 
                       'Graduation_Semester', 'course_category', 'time_bin', 'course_level']
    drop_cols = categorical_cols + ['minute']  # Drop minute as we have time_of_day_minutes
    
    # Keep only numeric columns and engineered features
    numeric_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns
    X = df_processed[numeric_cols].drop(columns=[target] if target in numeric_cols else [])
    
    # Add any one-hot encoded columns we created
    one_hot_cols = [col for col in df_processed.columns if col.startswith('hour_block_') or col.startswith('day_')]
    for col in one_hot_cols:
        if col in df_processed.columns:
            X[col] = df_processed[col]
    
    # Check for missing values in features
    missing_in_features = X.isnull().sum()
    logger.info(f"Missing values in features before imputation:\n{missing_in_features[missing_in_features > 0]}")
    
    # Handle missing values using imputation
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Verify missing values are handled
    logger.info(f"Missing values after imputation: {X_imputed.isnull().sum().sum()}")
    
    logger.info(f"Features after preprocessing: {X_imputed.columns.tolist()}")
    y = df_processed[target]
    
    # Save the imputer for prediction
    joblib.dump(imputer, f'{args.output_dir}/imputer_occupancy.pkl')
    logger.info(f"Imputer saved to {args.output_dir}/imputer_occupancy.pkl")
    
    # Get feature names for importance plots
    feature_names = X_imputed.columns.tolist()
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
    
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    
    # Train and evaluate multiple models
    best_model, best_model_name, results = train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names)
    
    # Tune hyperparameters if requested
    if args.tune and best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']:
        logger.info(f"Tuning hyperparameters for {best_model_name}...")
        tuned_model = tune_hyperparameters(X_train, y_train, best_model_name)
        if tuned_model is not None:
            # Evaluate the tuned model
            logger.info("Evaluating tuned model...")
            tuned_results = evaluate_model(tuned_model, X_test, y_test)
            logger.info(f"Tuned model - RMSE: {tuned_results['rmse']:.4f}, MAE: {tuned_results['mae']:.4f}, R²: {tuned_results['r2']:.4f}")
            
            # Visualize tuned model results
            visualize_results(y_test, tuned_results['predictions'], output_dir="occupancy_prediction/visualizations")
            
            # Generate feature importance plot for the tuned model
            feature_importance_plot(tuned_model, feature_names)
            
            # Use the tuned model as the best model
            best_model = tuned_model
    
    # Save the best model
    model_filename = f'{args.output_dir}/best_occupancy_predictor.pkl'
    joblib.dump(best_model, model_filename)
    logger.info(f"Best model saved to {model_filename}")
    
    # Visualize results for the best model
    best_model_results = results[best_model_name]
    visualize_results(y_test, best_model_results['predictions'], output_dir="occupancy_prediction/visualizations")
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': best_model_results['predictions']
    })
    predictions_df.to_csv(f'{args.output_dir}/model_predictions.csv', index=False)
    logger.info(f"Predictions saved to {args.output_dir}/model_predictions.csv")
    
    # Save model comparison results to the output directory
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'RMSE': [results[model]['rmse'] for model in results],
        'MAE': [results[model]['mae'] for model in results],
        'R²': [results[model]['r2'] for model in results],
        'Training Time (s)': [results[model]['training_time'] for model in results]
    })
    results_df.to_csv(f'{args.output_dir}/model_comparison_occupancy.csv', index=False)
    logger.info(f"Model comparison results saved to {args.output_dir}/model_comparison_occupancy.csv")
    
    # Print final results
    logger.info("\nFinal Results:")
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"RMSE: {best_model_results['rmse']:.4f}")
    logger.info(f"MAE: {best_model_results['mae']:.4f}")
    logger.info(f"R²: {best_model_results['r2']:.4f}")

if __name__ == '__main__':
    main() 