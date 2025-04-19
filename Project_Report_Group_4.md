### 3.3 Model Comparison Results

The final comparison of all models shows the following performance metrics:

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| XGBoost | 54.34 | 37.49 | 0.108 |
| Weighted Ensemble | 54.53 | 37.53 | 0.102 |
| Stacking Ensemble | 60.19 | 40.24 | 0.042 |

Key findings:
1. XGBoost remains the best performing model with an RMSE of 54.34 and R² of 0.108
2. The Weighted Ensemble approach shows competitive performance, with RMSE of 54.53 and R² of 0.102
3. The Stacking Ensemble model shows lower performance with RMSE of 60.19 and R² of 0.042

The weighted ensemble model combines predictions from four base models with the following weights:
- Random Forest: 25.23%
- LightGBM: 25.16%
- XGBoost: 24.83%
- Gradient Boosting: 24.78%

The balanced distribution of weights suggests that all models contribute similarly to the ensemble's performance, with Random Forest having a slightly higher influence. 