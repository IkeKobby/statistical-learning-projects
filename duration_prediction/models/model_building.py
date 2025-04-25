import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, cross_validate
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
import warnings
import os
from typing import Dict, Tuple, List, Any
warnings.filterwarnings('ignore')

# Load the engineered data directly
print("Loading engineered data...")
try:
    data = pd.read_csv('duration_prediction/processed_data/engineered_data.csv')
    print("Successfully loaded data from duration_prediction/processed_data/engineered_data.csv")
except FileNotFoundError:
    try:
        data = pd.read_csv('processed_data/engineered_data.csv')
        print("Successfully loaded data from processed_data/engineered_data.csv")
    except FileNotFoundError:
        try:
            data = pd.read_csv('../processed_data/engineered_data.csv')
            print("Successfully loaded data from ../processed_data/engineered_data.csv")
        except FileNotFoundError:
            print("Error: Could not find the engineered_data.csv file in any of the expected locations.")
            exit(1)

# Display basic information
print("\nDataset shape:", data.shape)
print("\nFirst few rows:")
print(data.head())

# Check for missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# Data types
print("\nData types:")
print(data.dtypes)

# Summary statistics for numerical variables
print("\nSummary statistics for numerical features:")
print(data.describe())

# Create directories for visualizations if they don't exist
os.makedirs('duration_prediction/visualizations', exist_ok=True)

# Distribution of target variable
plt.figure(figsize=(10, 6))
sns.histplot(data['Duration_In_Min'], kde=True)
plt.title('Distribution of Duration_In_Min')
plt.savefig('duration_prediction/visualizations/duration_distribution.png')
plt.close()

# Check for correlation between numerical features
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = data[numerical_cols].corr()
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.savefig('duration_prediction/visualizations/correlation_matrix.png')
plt.close()

# Calculate correlation with target variable
target_correlation = correlation_matrix['Duration_In_Min'].sort_values(ascending=False)
print("\nFeatures correlation with Duration_In_Min:")
print(target_correlation)

# Prepare for modeling
X = data.drop('Duration_In_Min', axis=1)
y = data['Duration_In_Min']

# -------------------------------------------------------------------------
# Model Building
# -------------------------------------------------------------------------
print("\n\nBuilding models...")

def evaluate_model(model, X, y, cv=5):
    """Evaluate a model using cross-validation."""
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    rmse_scores = -cv_scores
    mae_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    mae_scores = -mae_scores
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    
    return {
        'model': model.__class__.__name__,
        'rmse_mean': rmse_scores.mean(),
        'rmse_std': rmse_scores.std(),
        'mae_mean': mae_scores.mean(),
        'mae_std': mae_scores.std(),
        'r2_mean': r2_scores.mean(),
        'r2_std': r2_scores.std()
    }

def build_and_evaluate_models(X: pd.DataFrame, 
                           y: pd.Series,
                           test_size: float = 0.2,
                           random_state: int = 42,
                           save_results: bool = True,
                           results_path: str = 'duration_prediction/models/results/model_results.csv') -> Tuple[pd.DataFrame, ColumnTransformer]:
    """
    Build and evaluate multiple regression models for duration prediction.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    test_size : float, optional (default=0.2)
        Proportion of dataset to include in the test split
    random_state : int, optional (default=42)
        Random state for reproducibility
    save_results : bool, optional (default=True)
        Whether to save results to a CSV file
    results_path : str, optional (default='model_results.csv')
        Path where to save the results CSV file
        
    Returns:
    --------
    Tuple[pd.DataFrame, ColumnTransformer]
        - DataFrame containing model evaluation results
        - Fitted preprocessor for future use
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Define models with their hyperparameters
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(random_state=random_state),
        'Lasso': Lasso(random_state=random_state),
        'ElasticNet': ElasticNet(random_state=random_state),
        'Decision Tree': DecisionTreeRegressor(random_state=random_state),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
        'Gradient Boosting': GradientBoostingRegressor(random_state=random_state),
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'SVR': SVR(kernel='rbf'),
        'XGBoost': xgb.XGBRegressor(random_state=random_state),
        'LightGBM': lgb.LGBMRegressor(random_state=random_state, verbose=-1),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), 
                                     max_iter=1000, 
                                     random_state=random_state)
    }
    
    # Prepare results storage
    results = []
    
    # Fit preprocessor
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Convert to DataFrame if needed
    if isinstance(X_train_processed, np.ndarray):
        X_train_processed = pd.DataFrame(X_train_processed)
        X_test_processed = pd.DataFrame(X_test_processed)
    
    # Evaluate each model
    for name, model in models.items():
        try:
            print(f"Training {name}...")
            
            # Fit model
            model.fit(X_train_processed, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train_processed)
            y_pred_test = model.predict(X_test_processed)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_rmse = np.sqrt(train_mse)
            test_rmse = np.sqrt(test_mse)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Perform cross-validation
            cv_scores = cross_validate(model, X_train_processed, y_train,
                                    cv=5,
                                    scoring=['neg_mean_squared_error', 'r2'],
                                    n_jobs=-1)
            
            cv_rmse = np.sqrt(-cv_scores['test_neg_mean_squared_error'].mean())
            cv_r2 = cv_scores['test_r2'].mean()
            
            # Store results
            results.append({
                'Model': name,
                'Train RMSE': train_rmse,
                'Test RMSE': test_rmse,
                'CV RMSE': cv_rmse,
                'Train R2': train_r2,
                'Test R2': test_r2,
                'CV R2': cv_r2,
                'Train MAE': train_mae,
                'Test MAE': test_mae
            })
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by Test RMSE
    results_df = results_df.sort_values('Test RMSE')
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    # Save results if requested
    if save_results:
        results_df.to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")
    
    return results_df, preprocessor

# Split data for training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and evaluate models
print("\nBuilding and evaluating models...")
results_df, preprocessor = build_and_evaluate_models(X_train, y_train)

# -------------------------------------------------------------------------
# Fine-tune the best model
# -------------------------------------------------------------------------
print("\n\nFine-tuning the best model...")

# Get the best model name from the results
best_model_name = results_df.iloc[0]['Model']
print(f"\nBest performing model: {best_model_name}")

# Choose the best performing model for fine-tuning
if best_model_name == 'XGBoost':
    base_model = xgb.XGBRegressor()
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [3, 5, 7],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__subsample': [0.8, 0.9, 1.0],
        'model__colsample_bytree': [0.8, 0.9, 1.0],
        'model__min_child_weight': [1, 3, 5]
    }
elif best_model_name == 'Random Forest':
    base_model = RandomForestRegressor()
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }
elif best_model_name == 'Gradient Boosting':
    base_model = GradientBoostingRegressor()
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 5, 7],
        'model__min_samples_split': [2, 5, 10]
    }
elif best_model_name == 'LightGBM':
    base_model = lgb.LGBMRegressor(verbose=-1)
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__num_leaves': [31, 50, 100],
        'model__max_depth': [3, 5, 7]
    }
else:
    print(f"\nNo specific hyperparameter tuning implemented for {best_model_name}")
    print("Using default parameters for the best model")
    # Get the model from the previous dictionary
    model_dict = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42),
        'ElasticNet': ElasticNet(random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'SVR': SVR(kernel='rbf'),
        'XGBoost': xgb.XGBRegressor(random_state=42),
        'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    base_model = model_dict.get(best_model_name)
    param_grid = {}

# Create pipeline with preprocessor and base model
best_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', base_model)
])

# Use GridSearchCV for hyperparameter tuning if parameters are defined
if param_grid:
    print("\nPerforming hyperparameter tuning...")
    grid_search = GridSearchCV(
        best_model,
        param_grid,
        cv=5,  # Reduced from 10 to 5 for faster execution
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    # Fit grid search
    print("Running grid search (this may take some time)...")
    grid_search.fit(X_train, y_train)

    # Print best parameters
    print("\nBest parameters:")
    print(grid_search.best_params_)

    # Get the best model
    best_model = grid_search.best_estimator_
else:
    print("\nFitting the best model with default parameters...")
    best_model.fit(X_train, y_train)

# Evaluate best model on validation set
y_pred = best_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("\nBest model performance on validation set:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R^2: {r2:.4f}")

# -------------------------------------------------------------------------
# Feature Importance Analysis
# -------------------------------------------------------------------------
print("\n\nAnalyzing feature importance...")

# Try to extract feature importance
try:
    # Get feature names from preprocessor
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(cols)
        elif name == 'cat':
            for col in cols:
                try:
                    # Get the categories for each categorical column
                    categories = getattr(trans.named_steps['onehot'], 'categories_', None)
                    if categories:
                        for i, cat_col in enumerate(categories):
                            if i < len(cols):
                                feature_names.extend([f"{cols[i]}_{cat}" for cat in cat_col[1:]])  # Skip first category due to drop='first'
                except (AttributeError, IndexError) as e:
                    print(f"Error getting categories for {col}: {e}")
                    feature_names.append(f"{col}_unknown")

    # For tree-based models, extract feature importance
    if hasattr(best_model.named_steps['model'], 'feature_importances_'):
        importance = best_model.named_steps['model'].feature_importances_
        if len(feature_names) >= len(importance):
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(importance)],
                'importance': importance
            })
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # Plot top 20 features
            num_features_to_plot = min(20, len(importance_df))
            plt.figure(figsize=(12, 8))
            sns.barplot(data=importance_df.head(num_features_to_plot), x='importance', y='feature')
            plt.title(f'Top {num_features_to_plot} Feature Importances')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.savefig('duration_prediction/visualizations/feature_importance.png')
            plt.close()
            
            print(f"\nTop {num_features_to_plot} important features:")
            print(importance_df.head(num_features_to_plot))
        else:
            print("Number of importance values doesn't match feature names. Cannot create feature importance plot.")
    else:
        print("Model doesn't have feature_importances_ attribute. Cannot extract feature importance.")
except Exception as e:
    print(f"Error extracting feature importance: {e}")

# Plot predictions vs actual
try:
    plt.figure(figsize=(10, 8))
    plt.scatter(y_val, y_pred, alpha=0.5)
    plt.xlabel('Actual Duration')
    plt.ylabel('Predicted Duration')
    plt.title('Predictions vs Actual Values')
    plt.plot([0, y_val.max()], [0, y_val.max()], 'r--')
    plt.savefig('duration_prediction/visualizations/predictions_vs_actual.png')
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs actual plot: {e}")

# -------------------------------------------------------------------------
# Save the Best Model
# -------------------------------------------------------------------------
import joblib
# Create directory if it doesn't exist
os.makedirs('duration_prediction/models/saved_models', exist_ok=True)
joblib.dump(best_model, 'duration_prediction/models/saved_models/best_duration_predictor.pkl')
print("\nBest model saved as 'duration_prediction/models/saved_models/best_duration_predictor.pkl'")

print("\nAnalysis complete!") 
