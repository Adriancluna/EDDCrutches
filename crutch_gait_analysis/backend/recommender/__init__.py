"""
Recommender module for crutch fitting predictions.

Provides machine learning models that predict optimal initial crutch
settings based on user body measurements, and rule-based optimizers
for live adjustment recommendations.
"""

from .knn_recommender import KNNRecommender
from .live_optimizer import LiveFitOptimizer

__all__ = ['KNNRecommender', 'LiveFitOptimizer']
