"""
Learning Center Duration Prediction Package

This package provides tools for predicting duration in minutes for Learning Center visits.
"""

from .models.model_building import build_and_evaluate_models
from .models.stacking_model import create_stacking_model
from .models.Feature_engineer import train_advanced_models
from .utils.data_preprocessing import preprocess_data

__version__ = '0.1.0'
__all__ = [
    'build_and_evaluate_models',
    'create_stacking_model',
    'train_advanced_models',
    'preprocess_data'
] 