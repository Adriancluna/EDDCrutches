"""
Configuration module for crutch gait analysis.

Contains:
- clinical_guidelines: Clinical parameters and formulas for crutch fitting
- crutch_models: Device profiles with discrete adjustment settings
"""

from .clinical_guidelines import (
    ANTHROPOMETRIC_RATIOS,
    CRUTCH_FIT_FORMULAS,
    BIOMECHANICS_TARGETS,
    PENALTY_WEIGHTS,
    estimate_ideal_grip_height,
    estimate_ideal_overall_length,
    calculate_fit_penalty,
    get_fit_recommendation,
    validate_crutch_settings,
)

from .crutch_models import (
    CrutchDeviceProfile,
    DEFAULT_CRUTCH,
    TALL_CRUTCH,
    PEDIATRIC_CRUTCH,
    CRUTCH_PROFILES,
    get_profile,
)

__all__ = [
    # Clinical guidelines
    'ANTHROPOMETRIC_RATIOS',
    'CRUTCH_FIT_FORMULAS',
    'BIOMECHANICS_TARGETS',
    'PENALTY_WEIGHTS',
    'estimate_ideal_grip_height',
    'estimate_ideal_overall_length',
    'calculate_fit_penalty',
    'get_fit_recommendation',
    'validate_crutch_settings',
    # Crutch models
    'CrutchDeviceProfile',
    'DEFAULT_CRUTCH',
    'TALL_CRUTCH',
    'PEDIATRIC_CRUTCH',
    'CRUTCH_PROFILES',
    'get_profile',
]
