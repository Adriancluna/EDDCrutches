"""
Core module for crutch gait analysis.

Contains:
- geometry: Geometric calculations (2D and 3D angles, distances)
- evaluators: Phase-aware evaluation functions with clinical thresholds
- gait_detector: Gait phase detection
- temporal_tracker: Temporal smoothing for measurements
- one_euro_filter: Adaptive signal smoothing for landmarks
"""

from .geometry import (
    compute_distance,
    compute_distance_3d,
    compute_angle,
    compute_angle_3d,
    compute_angle_from_vertical,
    compute_angle_from_vertical_3d,
    check_visibility,
    get_visibility_mask,
    extract_point_2d,
    extract_point_3d,
    PoseGeometry,
    DEFAULT_VISIBILITY_THRESHOLD
)

from .evaluators import (
    evaluate_elbow,
    evaluate_knee,
    evaluate_trunk_lean,
    evaluate_shoulder_asym,
    evaluate_step_length,
    evaluate_base_width,
    calculate_step_ratio,
    calculate_base_ratio,
    THRESHOLDS
)

from .gait_detector import GaitPhaseDetector

from .temporal_tracker import TemporalTracker

from .one_euro_filter import (
    OneEuroFilter,
    LandmarkFilter,
    create_smooth_filter,
    create_responsive_filter,
    create_balanced_filter
)

from .crutch_detector import CrutchDetector

__all__ = [
    # Geometry
    'compute_distance',
    'compute_distance_3d',
    'compute_angle',
    'compute_angle_3d',
    'compute_angle_from_vertical',
    'compute_angle_from_vertical_3d',
    'check_visibility',
    'get_visibility_mask',
    'extract_point_2d',
    'extract_point_3d',
    'PoseGeometry',
    'DEFAULT_VISIBILITY_THRESHOLD',
    # Evaluators
    'evaluate_elbow',
    'evaluate_knee',
    'evaluate_trunk_lean',
    'evaluate_shoulder_asym',
    'evaluate_step_length',
    'evaluate_base_width',
    'calculate_step_ratio',
    'calculate_base_ratio',
    'THRESHOLDS',
    # Gait detection
    'GaitPhaseDetector',
    # Temporal tracking
    'TemporalTracker',
    # Filtering
    'OneEuroFilter',
    'LandmarkFilter',
    'create_smooth_filter',
    'create_responsive_filter',
    'create_balanced_filter',
    # Crutch detection
    'CrutchDetector',
]
