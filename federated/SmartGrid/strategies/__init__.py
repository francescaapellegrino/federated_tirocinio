"""
Balanced Aggregation Strategies for SmartGrid Federated Learning

This module provides optimized aggregation strategies for federated learning
with non-IID data distributions, specifically designed for SmartGrid datasets.

Available strategies:
- BalancedFedAvg: Main balanced aggregation strategy
- Class-weighted aggregation
- Outlier penalty methods
- Adaptive learning rates
- Hybrid approaches

Author: francescaapellegrino
Date: 2025-08-17
"""

from .balanced_strategy import (
    BalancedFedAvg,
    create_class_weighted_strategy,
    create_outlier_penalty_strategy,
    create_adaptive_strategy,
    create_hybrid_strategy,
    create_smartgrid_optimized_strategy
)

__all__ = [
    'BalancedFedAvg',
    'create_class_weighted_strategy',
    'create_outlier_penalty_strategy',
    'create_adaptive_strategy',
    'create_hybrid_strategy',
    'create_smartgrid_optimized_strategy'
]