import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
import joblib
from sklearn.impute import SimpleImputer
import warnings
import sys
import os
warnings.filterwarnings('ignore')

def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Create a preprocessor for the data.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
        
    Returns:
    --------
    ColumnTransformer
        Fitted preprocessor
    """
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

def build_stacking_model(X: pd.DataFrame, y: pd.Series, cv: int = 5) -> StackingRegressor:
    """
    Build a stacking ensemble model for duration prediction.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    cv : int, optional (default=5)
        Number of cross-validation folds
        
    Returns:
    --------
    StackingRegressor
        Fitted stacking ensemble model
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create preprocessor
    preprocessor = create_preprocessor(X)
    
    # Define base models
    base_models = [
        ('xgb', xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)),
        ('lgb', lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
        ('gbr', GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42))
    ]
    
    # Define meta-model
    meta_model = Ridge(alpha=1.0)
    
    # Create stacking model
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=cv,
        n_jobs=-1
    )
    
    # Create full pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', stacking_model)
    ])
    
    # Fit the model
    model.fit(X_train, y_train)
    
    return model

def evaluate_stacking_model(model: StackingRegressor, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> dict:
    """
    Evaluate the stacking model using cross-validation.
    
    Parameters:
    -----------
    model : StackingRegressor
        Fitted stacking model
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    cv : int, optional (default=5)
        Number of cross-validation folds
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Perform cross-validation
    rmse_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    mae_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    
    return {
        'rmse_mean': rmse_scores.mean(),
        'rmse_std': rmse_scores.std(),
        'mae_mean': mae_scores.mean(),
        'mae_std': mae_scores.std(),
        'r2_mean': r2_scores.mean(),
        'r2_std': r2_scores.std()
    }

print("Loading and preparing data...")
# Load data
try:
    data = pd.read_csv('duration_prediction/processed_data/engineered_data.csv')
except FileNotFoundError:
    print("File not found. Checking alternative paths...")
    try:
        # Try with relative path
        data = pd.read_csv('processed_data/engineered_data.csv')
    except FileNotFoundError:
        try:
            # Try with relative path from within duration_prediction
            data = pd.read_csv('../processed_data/engineered_data.csv')
        except FileNotFoundError:
            print("File not found in expected locations. Please check the data path.")
            sys.exit(1)

# Prepare for modeling
X = data.drop('Duration_In_Min', axis=1)
y = data['Duration_In_Min']

# Create preprocessor
preprocessor = create_preprocessor(X)

# Build and evaluate stacking model
print("\nBuilding stacking model...")
model = build_stacking_model(X, y)
results = evaluate_stacking_model(model, X, y)

# Print results
print("\nStacking Model Performance:")
print(f"RMSE: {results['rmse_mean']:.4f} (±{results['rmse_std']:.4f})")
print(f"MAE: {results['mae_mean']:.4f} (±{results['mae_std']:.4f})")
print(f"R²: {results['r2_mean']:.4f} (±{results['r2_std']:.4f})")

print("\nSaving model...")
# Save the model
try:
    os.makedirs('duration_prediction/models/saved_models', exist_ok=True)
    joblib.dump(model, 'duration_prediction/models/saved_models/stacking_model.pkl')
    print("Model saved successfully!")
except Exception as e:
    print(f"Error saving model: {e}")

# -------------------------------------------------------------------------
# Weighted Ensemble Approach
# -------------------------------------------------------------------------
print("\n\nBuilding Weighted Ensemble model...")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train individual models
print("Training individual models...")

models = {
    'xgb': Pipeline([
        ('preprocessor', preprocessor),
        ('model', xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42))
    ]),
    'lgb': Pipeline([
        ('preprocessor', preprocessor),
        ('model', lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1))
    ]),
    'rf': Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
    ]),
    'gbr': Pipeline([
        ('preprocessor', preprocessor),
        ('model', GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42))
    ])
}

# Train each model and store predictions
model_predictions = {}
model_weights = {}

# Split validation set from training set
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_sub, y_train_sub)
    # Get predictions on validation set
    val_preds = model.predict(X_val)
    # Calculate error
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"{name} validation RMSE: {val_rmse:.4f}")
    # Store predictions for test set
    model_predictions[name] = model.predict(X_test)
    # Higher weight for better models (lower RMSE)
    model_weights[name] = 1 / val_rmse

# Normalize weights to sum to 1
total_weight = sum(model_weights.values())
model_weights = {k: v/total_weight for k, v in model_weights.items()}

print("\nModel weights:")
for name, weight in model_weights.items():
    print(f"{name}: {weight:.4f}")

# Create weighted ensemble prediction
weighted_pred = np.zeros_like(y_test, dtype=float)
for name, preds in model_predictions.items():
    weighted_pred += preds * model_weights[name]

# Evaluate weighted ensemble
weighted_rmse = np.sqrt(mean_squared_error(y_test, weighted_pred))
weighted_mae = mean_absolute_error(y_test, weighted_pred)
weighted_r2 = r2_score(y_test, weighted_pred)

print("\nWeighted Ensemble performance on test set:")
print(f"RMSE: {weighted_rmse:.4f}")
print(f"MAE: {weighted_mae:.4f}")
print(f"R^2: {weighted_r2:.4f}")

# Save the weighted ensemble model
weighted_ensemble = {
    'models': models,
    'weights': model_weights
}
try:
    os.makedirs('duration_prediction/models/saved_models', exist_ok=True)
    joblib.dump(weighted_ensemble, 'duration_prediction/models/saved_models/weighted_ensemble.pkl')
    print("\nWeighted ensemble model saved successfully!")
except Exception as e:
    print(f"Error saving weighted ensemble model: {e}")

# -------------------------------------------------------------------------
# Compare all models 
# -------------------------------------------------------------------------
print("\n\nComparing all models...")

# Create comparison DataFrame
comparison_results = pd.DataFrame({
    'Stacking Ensemble': {
        'RMSE': results['rmse_mean'],
        'MAE': results['mae_mean'],
        'R²': results['r2_mean']
    },
    'Weighted Ensemble': {
        'RMSE': weighted_rmse,
        'MAE': weighted_mae,
        'R²': weighted_r2
    }
}).T

# Try to load other model results
try:
    best_model = joblib.load('duration_prediction/models/saved_models/best_duration_predictor.pkl')
    y_pred_best = best_model.predict(X_test)
    comparison_results.loc['Best Individual Model (LightGBM)'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_best)),
        'MAE': mean_absolute_error(y_test, y_pred_best),
        'R²': r2_score(y_test, y_pred_best)
    }
except Exception as e:
    print(f"Could not load best model: {e}")

# Sort by RMSE
comparison_results = comparison_results.sort_values('RMSE')

print("\nFinal Model Comparison:")
print(comparison_results)

# Create visualizations directory if it doesn't exist
os.makedirs('duration_prediction/visualizations', exist_ok=True)

# Plot comparison
try:
    plt.figure(figsize=(10, 6))
    comparison_results['RMSE'].plot(kind='bar', color='skyblue')
    plt.title('RMSE Comparison Across All Models')
    plt.ylabel('RMSE (lower is better)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('duration_prediction/visualizations/model_comparison.png')
    plt.close()
    print("Model comparison visualization saved to duration_prediction/visualizations/model_comparison.png")
except Exception as e:
    print(f"Error creating model comparison visualization: {e}")

print("\nStacking and Weighted Ensemble Modeling Analysis complete!") 