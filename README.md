# Duration Prediction Models

This repository contains multiple machine learning models for predicting `Duration_In_Min` in the provided dataset. The goal is to build accurate predictive models that can be ranked in the top 1% of the leaderboard in the class competition.

## Project Structure

- `model_building.py`: Main script for exploratory data analysis, feature engineering, and building multiple machine learning models
- `advanced_models.py`: Script for implementing advanced models including neural networks and feature selection approaches
- `stacking_model.py`: Script for building ensemble models using stacking and weighted averaging techniques
- `predict.py`: Script for making predictions using the trained models
- `*.pkl`: Saved model files after training
- `*.png`: Generated plots and visualizations

## Models Implemented

We've built 5 different types of models:

1. **XGBoost**: An optimized gradient boosting model with hyperparameter tuning
2. **Neural Network**: A deep learning model with multiple layers, batch normalization, and dropout
3. **Feature Selection + Linear Model**: A hybrid approach using XGBoost for feature selection and linear regression for prediction
4. **Lasso Regression**: A linear model with L1 regularization and hyperparameter tuning
5. **Ensemble Models**: 
   - Stacking ensemble using multiple base models and a meta-learner
   - Weighted average ensemble that combines predictions from multiple models

## Feature Engineering

We created multiple engineered features to improve model performance:

- Time-related features (time of day in minutes, time to noon, peak hours)
- Academic features (progress ratio, student engagement)
- Course-related features (course complexity, STEM indicators)
- Cyclical features (for days of week, hours)
- Student progress indicators (years to graduation, semester progress)
- Interaction features between various original features

## Usage

### Running the Exploratory Analysis and Initial Models

```bash
python model_building.py
```

This will:
- Load and explore the dataset
- Apply feature engineering
- Train and evaluate multiple ML models
- Fine-tune the best model
- Save the best model as `best_duration_predictor.pkl`

### Running Advanced Models

```bash
python advanced_models.py
```

This will:
- Load the engineered data or recreate it
- Build and train a neural network model
- Implement feature selection with a linear model
- Build a Lasso regression model with hyperparameter tuning
- Compare all models and save results

### Running Ensemble Models

```bash
python stacking_model.py
```

This will:
- Build a stacking ensemble model
- Build a weighted ensemble model
- Compare with previous models
- Save the ensemble models

### Making Predictions

```bash
python predict.py --data test_data.csv --model stacking_ensemble_model.pkl --output predictions.csv
```

Options:
- `--data`: Path to the test data file (required)
- `--model`: Model to use for predictions (default: stacking_ensemble_model.pkl)
- `--output`: Path to save predictions (default: predictions.csv)

## Model Performance

The models were evaluated using cross-validation and test set performance metrics:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (R-squared)

The best performing model was the Stacking Ensemble, which combines the strengths of multiple models.

## Data Features

The dataset includes the following feature categories:

- Student demographics and academic standing
- Course information
- Temporal features (time of day, day of week)
- Academic performance metrics
- Engineered features for enhanced prediction 