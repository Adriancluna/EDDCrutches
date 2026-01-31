"""
Phase-Aware Evaluation Functions & Clinical Thresholds

Evaluation rules that adapt based on gait phase.
Supports both absolute measurements and normalized ratios.
"""

# ========================================
# PHASE-SPECIFIC THRESHOLDS
# ========================================

THRESHOLDS = {
    # STANDING: Relaxed - no weight on crutches
    'standing': {
        'elbow': {
            'optimal_min': 155,
            'optimal_max': 180,
            'warning_min': 145,
            'critical_min': 120
        },
        'trunk_lean': {
            'optimal_max': 20,
            'warning_max': 25,
            'critical_max': 30
        }
    },

    # WEIGHT_BEARING: Strict - supporting body weight
    # Adjusted based on clinical guidelines for crutch use
    'weight_bearing': {
        'elbow': {
            'critical_max': 180,      # Fully locked is bad
            'warning_max': 172,       # Close to locked
            'optimal_max': 168,       # Slight bend acceptable
            'optimal_min': 145,       # 20-30 degrees flexion
            'warning_min': 135,       # Too bent
            'critical_min': 115       # Way too bent
        },
        'trunk_lean': {
            'optimal_max': 18,        # Slight forward lean OK
            'warning_max': 25,        # Moderate lean
            'critical_max': 35        # Excessive lean
        }
    },

    # SWING/TRANSITION: Moderate thresholds
    'swing': {
        'elbow': {
            'optimal_min': 150,
            'optimal_max': 175,
            'warning_min': 140,
            'critical_min': 120
        },
        'trunk_lean': {
            'optimal_max': 20,
            'warning_max': 28,
            'critical_max': 35
        }
    },

    # Phase-independent thresholds
    'knee_stance': {
        'warning_min': 155,
        'optimal_min': 165,
        'optimal_max': 178,
        'warning_max': 180
    },
    'shoulder_asym': {
        'optimal_max': 4,
        'warning_max': 7,
        'critical_max': 12
    },

    # ========================================
    # NORMALIZED RATIO THRESHOLDS (NEW)
    # ========================================
    # These use ratios relative to leg length for body-size independence

    # Step length as percentage of leg length
    # Normal gait: 50-80% of leg length
    # Crutch users typically have shorter steps: 30-60%
    'step_length_ratio': {
        'critical_min': 0.15,    # < 15% = shuffling
        'warning_min': 0.25,     # < 25% = very short
        'optimal_min': 0.35,     # 35% minimum
        'optimal_max': 0.70,     # 70% maximum
        'warning_max': 0.85,     # > 85% = overstriding
        'critical_max': 1.00     # > 100% = likely measurement error
    },

    # Base of support width as percentage of hip width
    # Normal: 50-120% of hip width
    # For stability with crutches: 60-100%
    'base_width_ratio': {
        'critical_min': 0.30,    # < 30% = too narrow
        'warning_min': 0.50,     # < 50% = narrow
        'optimal_min': 0.60,     # 60% minimum
        'optimal_max': 1.20,     # 120% maximum
        'warning_max': 1.50,     # > 150% = wide
        'critical_max': 2.00     # > 200% = very wide
    },

    # ========================================
    # LEGACY ABSOLUTE THRESHOLDS (cm)
    # Keep for backwards compatibility
    # ========================================
    'step_length': {
        'critical_min': 12,
        'warning_min': 18,
        'optimal_min': 25,
        'optimal_max': 55,
        'warning_max': 70,
        'critical_max': 90
    },
    'base_width': {
        'critical_min': 5,
        'warning_min': 10,
        'optimal_min': 15,
        'optimal_max': 35,
        'warning_max': 50,
        'critical_max': 65
    }
}


def evaluate_elbow(angle, side='right', gait_phase='standing'):
    """
    Evaluate elbow angle with phase-aware thresholds.

    Args:
        angle: Elbow angle in degrees (None = landmark not visible)
        side: 'right' or 'left'
        gait_phase: Current gait phase
    """
    # Handle missing data
    if angle is None:
        return {
            'status': 'unknown',
            'message': f"{side.upper()} elbow not visible",
            'color': (128, 128, 128)  # Gray
        }

    # Select appropriate thresholds based on phase
    if gait_phase in ['WEIGHT_BEARING_LEFT', 'WEIGHT_BEARING_RIGHT']:
        t = THRESHOLDS['weight_bearing']['elbow']
    elif gait_phase in ['SWING_PHASE', 'DOUBLE_SUPPORT']:
        t = THRESHOLDS['swing']['elbow']
    else:  # STANDING or unknown
        t = THRESHOLDS['standing']['elbow']

    # Evaluate based on selected thresholds
    if 'critical_max' in t and angle > t['critical_max']:
        return {
            'status': 'critical',
            'message': f"{side.upper()} elbow locked - bend slightly",
            'color': (0, 0, 255)
        }
    elif 'warning_max' in t and angle > t['warning_max']:
        return {
            'status': 'warning',
            'message': f"{side.upper()} elbow too straight",
            'color': (0, 165, 255)
        }
    elif angle < t['critical_min']:
        return {
            'status': 'critical',
            'message': f"{side.upper()} elbow too bent - lower arm",
            'color': (0, 0, 255)
        }
    elif angle < t['warning_min']:
        return {
            'status': 'warning',
            'message': f"{side.upper()} elbow too bent",
            'color': (0, 165, 255)
        }
    elif t['optimal_min'] <= angle <= t['optimal_max']:
        return {'status': 'good', 'message': None, 'color': (0, 255, 0)}
    else:
        return {'status': 'caution', 'message': None, 'color': (0, 255, 255)}


def evaluate_trunk_lean(angle, gait_phase='standing'):
    """
    Evaluate trunk lean with phase-aware thresholds.

    Args:
        angle: Trunk lean angle in degrees (None = landmark not visible)
        gait_phase: Current gait phase
    """
    # Handle missing data
    if angle is None:
        return {
            'status': 'unknown',
            'message': "Trunk not visible",
            'color': (128, 128, 128)
        }

    # Select thresholds
    if gait_phase in ['WEIGHT_BEARING_LEFT', 'WEIGHT_BEARING_RIGHT']:
        t = THRESHOLDS['weight_bearing']['trunk_lean']
    elif gait_phase in ['SWING_PHASE', 'DOUBLE_SUPPORT']:
        t = THRESHOLDS['swing']['trunk_lean']
    else:
        t = THRESHOLDS['standing']['trunk_lean']

    if angle > t['critical_max']:
        return {
            'status': 'critical',
            'message': "Leaning too far forward - stand upright",
            'color': (0, 0, 255)
        }
    elif angle > t['warning_max']:
        return {
            'status': 'warning',
            'message': "Excessive forward lean",
            'color': (0, 165, 255)
        }
    elif angle <= t['optimal_max']:
        return {'status': 'good', 'message': None, 'color': (0, 255, 0)}
    else:
        return {'status': 'caution', 'message': None, 'color': (0, 255, 255)}


def evaluate_knee(angle, side='right'):
    """
    Evaluate knee angle (phase-independent).

    Args:
        angle: Knee angle in degrees (None = landmark not visible)
        side: 'right' or 'left'
    """
    # Handle missing data
    if angle is None:
        return {
            'status': 'unknown',
            'message': f"{side.upper()} knee not visible",
            'color': (128, 128, 128)
        }

    t = THRESHOLDS['knee_stance']

    if angle < t['warning_min']:
        return {
            'status': 'warning',
            'message': f"{side.upper()} knee too bent",
            'color': (0, 165, 255)
        }
    elif angle > t['warning_max']:
        return {
            'status': 'warning',
            'message': f"{side.upper()} knee locked - allow slight bend",
            'color': (0, 165, 255)
        }
    elif t['optimal_min'] <= angle <= t['optimal_max']:
        return {'status': 'good', 'message': None, 'color': (0, 255, 0)}
    else:
        return {'status': 'caution', 'message': None, 'color': (0, 255, 255)}


def evaluate_shoulder_asym(percent):
    """
    Evaluate shoulder asymmetry.

    Args:
        percent: Shoulder height difference as percentage of leg length
                 (None = landmark not visible)
    """
    # Handle missing data
    if percent is None:
        return {
            'status': 'unknown',
            'message': "Shoulders not visible",
            'color': (128, 128, 128)
        }

    t = THRESHOLDS['shoulder_asym']

    if percent > t['critical_max']:
        return {
            'status': 'critical',
            'message': "Shoulders very uneven - check crutch height",
            'color': (0, 0, 255)
        }
    elif percent > t['warning_max']:
        return {
            'status': 'warning',
            'message': "Shoulders uneven",
            'color': (0, 165, 255)
        }
    elif percent <= t['optimal_max']:
        return {'status': 'good', 'message': None, 'color': (0, 255, 0)}
    else:
        return {'status': 'caution', 'message': None, 'color': (0, 255, 255)}


def evaluate_step_length(length_cm=None, ratio=None, use_ratio=True):
    """
    Evaluate step length using either absolute cm or normalized ratio.

    Args:
        length_cm: Step length in cm (legacy)
        ratio: Step length as ratio of leg length (preferred)
        use_ratio: If True and ratio provided, use ratio thresholds

    The ratio method is preferred because it accounts for body size.
    A 150cm person has a different "normal" step than a 190cm person.
    """
    # Prefer ratio if available
    if use_ratio and ratio is not None:
        return _evaluate_ratio(
            ratio,
            THRESHOLDS['step_length_ratio'],
            name="Step length",
            too_short_msg="Steps very short - try longer strides",
            too_long_msg="Steps too long - reduce stride"
        )

    # Fallback to absolute cm
    if length_cm is None:
        return {
            'status': 'unknown',
            'message': "Step length not measurable",
            'color': (128, 128, 128)
        }

    t = THRESHOLDS['step_length']

    if length_cm < t['critical_min']:
        return {
            'status': 'critical',
            'message': "Steps very short - try longer strides",
            'color': (0, 0, 255)
        }
    elif length_cm < t['warning_min']:
        return {
            'status': 'warning',
            'message': "Steps too short",
            'color': (0, 165, 255)
        }
    elif length_cm > t['critical_max']:
        return {
            'status': 'critical',
            'message': "Steps too long - reduce stride",
            'color': (0, 0, 255)
        }
    elif length_cm > t['warning_max']:
        return {
            'status': 'warning',
            'message': "Steps too long",
            'color': (0, 165, 255)
        }
    elif t['optimal_min'] <= length_cm <= t['optimal_max']:
        return {'status': 'good', 'message': None, 'color': (0, 255, 0)}
    else:
        return {'status': 'caution', 'message': None, 'color': (0, 255, 255)}


def evaluate_base_width(width_cm=None, ratio=None, use_ratio=True):
    """
    Evaluate base of support width using either absolute cm or normalized ratio.

    Args:
        width_cm: Base width in cm (legacy)
        ratio: Base width as ratio of hip width (preferred)
        use_ratio: If True and ratio provided, use ratio thresholds

    The ratio method is preferred because it accounts for body size.
    """
    # Prefer ratio if available
    if use_ratio and ratio is not None:
        return _evaluate_ratio(
            ratio,
            THRESHOLDS['base_width_ratio'],
            name="Base width",
            too_short_msg="Stance very narrow - widen for stability",
            too_long_msg="Stance very wide - bring feet closer"
        )

    # Fallback to absolute cm
    if width_cm is None:
        return {
            'status': 'unknown',
            'message': "Base width not measurable",
            'color': (128, 128, 128)
        }

    t = THRESHOLDS['base_width']

    if width_cm < t['critical_min']:
        return {
            'status': 'critical',
            'message': "Stance very narrow - widen for stability",
            'color': (0, 0, 255)
        }
    elif width_cm < t['warning_min']:
        return {
            'status': 'warning',
            'message': "Stance too narrow",
            'color': (0, 165, 255)
        }
    elif width_cm > t['critical_max']:
        return {
            'status': 'critical',
            'message': "Stance very wide - bring feet closer",
            'color': (0, 0, 255)
        }
    elif width_cm > t['warning_max']:
        return {
            'status': 'warning',
            'message': "Stance too wide",
            'color': (0, 165, 255)
        }
    elif t['optimal_min'] <= width_cm <= t['optimal_max']:
        return {'status': 'good', 'message': None, 'color': (0, 255, 0)}
    else:
        return {'status': 'caution', 'message': None, 'color': (0, 255, 255)}


def _evaluate_ratio(ratio, thresholds, name, too_short_msg, too_long_msg):
    """
    Generic ratio evaluation helper.

    Args:
        ratio: The ratio value to evaluate
        thresholds: Dict with critical/warning/optimal min/max
        name: Name for unknown message
        too_short_msg: Message for too small values
        too_long_msg: Message for too large values
    """
    if ratio is None:
        return {
            'status': 'unknown',
            'message': f"{name} not measurable",
            'color': (128, 128, 128)
        }

    t = thresholds

    if ratio < t['critical_min']:
        return {
            'status': 'critical',
            'message': too_short_msg,
            'color': (0, 0, 255)
        }
    elif ratio < t['warning_min']:
        return {
            'status': 'warning',
            'message': too_short_msg.replace("very", "too"),
            'color': (0, 165, 255)
        }
    elif ratio > t['critical_max']:
        return {
            'status': 'critical',
            'message': too_long_msg,
            'color': (0, 0, 255)
        }
    elif ratio > t['warning_max']:
        return {
            'status': 'warning',
            'message': too_long_msg.replace("very", "too"),
            'color': (0, 165, 255)
        }
    elif t['optimal_min'] <= ratio <= t['optimal_max']:
        return {'status': 'good', 'message': None, 'color': (0, 255, 0)}
    else:
        return {'status': 'caution', 'message': None, 'color': (0, 255, 255)}


def calculate_step_ratio(step_length_cm, leg_length_cm):
    """
    Calculate step length as ratio of leg length.

    Args:
        step_length_cm: Measured step length in cm
        leg_length_cm: Calibrated leg length in cm

    Returns:
        Ratio (0-1+) or None if invalid
    """
    if leg_length_cm is None or leg_length_cm < 10:
        return None
    if step_length_cm is None:
        return None
    return step_length_cm / leg_length_cm


def calculate_base_ratio(base_width_cm, hip_width_cm):
    """
    Calculate base of support width as ratio of hip width.

    Args:
        base_width_cm: Measured base width in cm
        hip_width_cm: Measured hip width in cm

    Returns:
        Ratio (0-2+) or None if invalid
    """
    if hip_width_cm is None or hip_width_cm < 5:
        return None
    if base_width_cm is None:
        return None
    return base_width_cm / hip_width_cm


print("Phase-aware evaluation functions loaded (with visibility gating & normalized ratios)")
