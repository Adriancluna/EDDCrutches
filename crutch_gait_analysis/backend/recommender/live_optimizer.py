#!/usr/bin/env python3
"""
Live Fit Optimizer for Real-Time Crutch Adjustments.

This module provides rule-based optimization that analyzes biomechanics
during gait and suggests specific crutch adjustments based on clinical
physical therapy best practices.

Clinical Rationale:
-------------------
The optimizer uses established relationships between observed biomechanics
and crutch fit:

1. Elbow Angle (target: 150-165°, ideal: 155°)
   - Too straight (>165°): User is reaching down → handle too high OR crutch too short
   - Too bent (<150°): User is hunched → handle too low

2. Trunk Lean (target: <12°, ideal: 5°)
   - Forward lean: Compensating for short crutch OR weak core
   - Combined with straight elbows: Strong indicator crutch is too short

3. Asymmetry
   - Large left/right elbow difference may indicate injury, pain avoidance,
     or uneven crutch heights. Should be flagged for clinical review.

Adjustment Strategy:
-------------------
- One adjustment at a time (isolate variables)
- Standard step: 1 inch (2.54 cm) = one hole position
- Re-evaluate after each adjustment
- Converge when score improvement < 0.5 points
"""

import numpy as np
from config.clinical_guidelines import BIOMECHANICS_TARGETS
from config.crutch_models import DEFAULT_CRUTCH


class LiveFitOptimizer:
    """
    Rule-based optimizer that suggests crutch adjustments based on
    observed biomechanics during live gait analysis.

    The optimizer follows a decision tree based on clinical guidelines:

    1. Check for asymmetry first (may indicate medical issue)
    2. Analyze elbow angles vs trunk lean to determine root cause
    3. Suggest single adjustment to isolate effect
    4. Track convergence via fit score

    Attributes:
        target_elbow_min: Minimum acceptable elbow angle (150°)
        target_elbow_max: Maximum acceptable elbow angle (165°)
        target_elbow_ideal: Ideal elbow angle (155°)
        target_trunk_max: Maximum acceptable trunk lean (12°)
        target_trunk_ideal: Ideal trunk lean (5°)
        adjustment_step_cm: Standard adjustment step (2.54cm = 1 inch)

    Example:
        >>> optimizer = LiveFitOptimizer()
        >>> biomech = optimizer.analyze_weight_bearing_frames(session_frames)
        >>> score = optimizer.compute_fit_score(biomech)
        >>> suggestions = optimizer.recommend_adjustment(biomech, current_config)
        >>> optimizer.print_suggestions(suggestions)
    """

    def __init__(self):
        """
        Initialize optimizer with clinical target values.

        Targets are loaded from BIOMECHANICS_TARGETS configuration
        which reflects physical therapy best practices.
        """
        self.target_elbow_min = BIOMECHANICS_TARGETS['elbow_angle_min']
        self.target_elbow_max = BIOMECHANICS_TARGETS['elbow_angle_max']
        self.target_elbow_ideal = BIOMECHANICS_TARGETS['elbow_angle_ideal']
        self.target_trunk_max = BIOMECHANICS_TARGETS['trunk_lean_max']
        self.target_trunk_ideal = BIOMECHANICS_TARGETS['trunk_lean_ideal']

        # Standard adjustment = 1 hole = 1 inch
        self.adjustment_step_cm = 2.54

    def analyze_weight_bearing_frames(self, frames):
        """
        Analyze weight-bearing frames to extract biomechanics patterns.

        Only weight-bearing phases are analyzed because this is when
        the user is actually loading the crutches and biomechanics
        measurements are most meaningful for fit assessment.

        Args:
            frames: List of frame dicts from session data, each containing:
                - measurements.elbow_r: Right elbow angle (degrees)
                - measurements.elbow_l: Left elbow angle (degrees)
                - measurements.trunk_lean: Forward trunk lean (degrees)
                - measurements.gait_phase: Current gait phase string

        Returns:
            dict with aggregated biomechanics:
                - avg_elbow_r: Mean right elbow angle
                - avg_elbow_l: Mean left elbow angle
                - avg_elbow: Mean of both elbows
                - avg_trunk: Mean trunk lean
                - std_elbow_r: Right elbow std dev (stability)
                - std_elbow_l: Left elbow std dev (stability)
                - n_frames: Number of frames analyzed
                - elbow_asymmetry: Absolute L/R difference

            Returns None if insufficient data (<20 frames)
        """
        # Filter to weight-bearing phases only
        wb_frames = [
            f for f in frames
            if f['measurements'].get('gait_phase', '').startswith('WEIGHT_BEARING')
        ]

        if len(wb_frames) < 20:
            return None  # Not enough data for reliable analysis

        # Calculate averages
        avg_elbow_r = np.mean([f['measurements']['elbow_r'] for f in wb_frames])
        avg_elbow_l = np.mean([f['measurements']['elbow_l'] for f in wb_frames])
        avg_trunk = np.mean([f['measurements']['trunk_lean'] for f in wb_frames])

        # Standard deviations indicate stability/consistency
        std_elbow_r = np.std([f['measurements']['elbow_r'] for f in wb_frames])
        std_elbow_l = np.std([f['measurements']['elbow_l'] for f in wb_frames])

        return {
            'avg_elbow_r': avg_elbow_r,
            'avg_elbow_l': avg_elbow_l,
            'avg_elbow': (avg_elbow_r + avg_elbow_l) / 2,
            'avg_trunk': avg_trunk,
            'std_elbow_r': std_elbow_r,
            'std_elbow_l': std_elbow_l,
            'n_frames': len(wb_frames),
            'elbow_asymmetry': abs(avg_elbow_r - avg_elbow_l)
        }

    def compute_fit_score(self, biomech):
        """
        Compute single numerical fit score (lower is better).

        The score uses a penalty system that:
        - Penalizes deviation from target ranges
        - Applies exponential penalty for critical deviations
        - Weights trunk lean slightly higher (affects balance/safety)

        Scoring breakdown:
        - Elbow outside range: 0.3 points per degree
        - Elbow critical (>10° off): Additional 0.5 points per degree
        - Trunk outside range: 0.4 points per degree
        - Trunk critical (>5° off): Additional 0.6 points per degree

        Args:
            biomech: Dict from analyze_weight_bearing_frames()

        Returns:
            float score (0 = perfect, higher = worse fit)
            Returns infinity if biomech is None
        """
        if biomech is None:
            return float('inf')

        score = 0.0

        # Elbow penalty (both sides independently)
        for side in ['avg_elbow_r', 'avg_elbow_l']:
            elbow = biomech[side]

            # Too bent (elbow angle too small)
            if elbow < self.target_elbow_min:
                deviation = self.target_elbow_min - elbow
                score += deviation * 0.3
                if deviation > 10:  # Critical deviation
                    score += deviation * 0.5  # Extra penalty

            # Too straight (elbow angle too large)
            elif elbow > self.target_elbow_max:
                deviation = elbow - self.target_elbow_max
                score += deviation * 0.3
                if deviation > 10:  # Critical deviation
                    score += deviation * 0.5

        # Trunk lean penalty
        trunk = biomech['avg_trunk']
        if trunk > self.target_trunk_max:
            deviation = trunk - self.target_trunk_max
            score += deviation * 0.4
            if deviation > 5:  # Critical forward lean
                score += deviation * 0.6

        return round(score, 2)

    def recommend_adjustment(self, biomech, current_config, device_profile=None):
        """
        Recommend next adjustment based on observed biomechanics.

        Uses clinical decision tree:

        1. ASYMMETRY CHECK (>10° L/R difference)
           → Warning only, don't auto-adjust (may be medical issue)

        2. ELBOWS TOO STRAIGHT (>170°)
           a. With forward lean → Crutch too short → Raise overall
           b. Without lean → Handle too high → Lower grip

        3. ELBOWS TOO BENT (<145°)
           → Handle too low → Raise grip

        4. EXCESSIVE TRUNK LEAN (>17°)
           → Crutch may be too short → Raise overall

        5. ALL GOOD
           → Return success message

        Args:
            biomech: Dict from analyze_weight_bearing_frames()
            current_config: Current crutch settings dict with:
                - grip_height_cm
                - grip_setting_idx
                - overall_length_cm
                - overall_setting_idx
            device_profile: CrutchDeviceProfile (defaults to DEFAULT_CRUTCH)

        Returns:
            list of suggestion dicts, each containing:
                - type: 'grip_height', 'overall_height', 'warning', or 'success'
                - adjustment: 'increase', 'decrease', or None
                - amount_cm: Adjustment amount (if applicable)
                - reason: Human-readable explanation
                - priority: 'high', 'medium', or 'info'

            Returns empty list if biomech is None
        """
        if biomech is None:
            return []

        if device_profile is None:
            device_profile = DEFAULT_CRUTCH

        suggestions = []

        avg_elbow = biomech['avg_elbow']
        avg_trunk = biomech['avg_trunk']
        asymmetry = biomech['elbow_asymmetry']

        # ================================================================
        # RULE 1: Check for asymmetry first
        # Large L/R difference may indicate injury, pain avoidance, or
        # uneven settings. Don't auto-adjust - flag for clinical review.
        # ================================================================
        if asymmetry > 10:
            suggestions.append({
                'type': 'warning',
                'adjustment': None,
                'amount_cm': None,
                'reason': f"Large elbow asymmetry ({asymmetry:.1f}°). "
                         f"Check for injury or uneven crutch heights.",
                'priority': 'high'
            })
            return suggestions  # Don't suggest other adjustments

        # ================================================================
        # RULE 2: Both elbows too straight (>170°)
        # User is reaching down to the handles, indicating either:
        # a) Handle is too high relative to body, OR
        # b) Entire crutch is too short (user reaching down and forward)
        # ================================================================
        if avg_elbow > self.target_elbow_max + 5:

            # Sub-rule 2a: Straight elbows + forward lean
            # This combination strongly suggests crutch is too SHORT
            # (user reaching down AND compensating by leaning forward)
            if avg_trunk > self.target_trunk_max:
                suggestions.append({
                    'type': 'overall_height',
                    'adjustment': 'increase',
                    'amount_cm': self.adjustment_step_cm,
                    'reason': f"Elbows too straight ({avg_elbow:.1f}°) + "
                             f"forward lean ({avg_trunk:.1f}°) suggests crutch too short",
                    'priority': 'high'
                })

            # Sub-rule 2b: Just straight elbows, trunk is OK
            # Handle position is too high relative to body
            else:
                suggestions.append({
                    'type': 'grip_height',
                    'adjustment': 'decrease',
                    'amount_cm': self.adjustment_step_cm,
                    'reason': f"Elbows too straight ({avg_elbow:.1f}°) → "
                             f"handle likely too high",
                    'priority': 'medium'
                })

        # ================================================================
        # RULE 3: Both elbows too bent (<145°)
        # User's hands are too high, forcing elbow flexion
        # Handle needs to be raised
        # ================================================================
        elif avg_elbow < self.target_elbow_min - 5:
            suggestions.append({
                'type': 'grip_height',
                'adjustment': 'increase',
                'amount_cm': self.adjustment_step_cm,
                'reason': f"Elbows too bent ({avg_elbow:.1f}°) → handle too low",
                'priority': 'medium'
            })

        # ================================================================
        # RULE 4: Excessive trunk lean (independent of elbows)
        # Forward lean may indicate crutch too short or weak core
        # Only add if not already suggested overall height change
        # ================================================================
        if avg_trunk > self.target_trunk_max + 5:
            if not any(s['type'] == 'overall_height' for s in suggestions):
                suggestions.append({
                    'type': 'overall_height',
                    'adjustment': 'increase',
                    'amount_cm': self.adjustment_step_cm,
                    'reason': f"Excessive forward lean ({avg_trunk:.1f}°)",
                    'priority': 'medium'
                })

        # ================================================================
        # RULE 5: Everything looks good!
        # Both elbows and trunk are within target ranges
        # ================================================================
        if not suggestions:
            if (self.target_elbow_min <= avg_elbow <= self.target_elbow_max and
                    avg_trunk <= self.target_trunk_max):
                suggestions.append({
                    'type': 'success',
                    'adjustment': None,
                    'amount_cm': None,
                    'reason': f"Fit is optimal! Elbow: {avg_elbow:.1f}°, "
                             f"Trunk: {avg_trunk:.1f}°",
                    'priority': 'info'
                })

        return suggestions

    def apply_suggestion(self, suggestion, current_config, device_profile=None):
        """
        Apply a suggestion to current config, returning new config.

        Converts the suggested cm adjustment to the nearest available
        device setting (crutches have discrete hole positions).

        Args:
            suggestion: Suggestion dict from recommend_adjustment()
            current_config: Current settings dict with grip/overall values
            device_profile: CrutchDeviceProfile for setting conversion

        Returns:
            new_config dict with updated settings
            (Returns copy of current_config if suggestion has no adjustment)
        """
        if device_profile is None:
            device_profile = DEFAULT_CRUTCH

        new_config = current_config.copy()

        if suggestion['type'] == 'grip_height' and suggestion['adjustment']:
            current_cm = current_config['grip_height_cm']

            if suggestion['adjustment'] == 'increase':
                new_cm = current_cm + suggestion['amount_cm']
            else:  # decrease
                new_cm = current_cm - suggestion['amount_cm']

            # Convert to nearest available setting
            info = device_profile.cm_to_setting(new_cm, 'grip')
            new_config['grip_height_cm'] = info['actual_cm']
            new_config['grip_setting_idx'] = info['setting_idx']

        elif suggestion['type'] == 'overall_height' and suggestion['adjustment']:
            current_cm = current_config['overall_length_cm']

            if suggestion['adjustment'] == 'increase':
                new_cm = current_cm + suggestion['amount_cm']
            else:
                new_cm = current_cm - suggestion['amount_cm']

            # Convert to nearest available setting
            info = device_profile.cm_to_setting(new_cm, 'overall')
            new_config['overall_length_cm'] = info['actual_cm']
            new_config['overall_setting_idx'] = info['setting_idx']

        return new_config

    def check_convergence(self, score_history, threshold=0.5):
        """
        Check if optimization has converged.

        Convergence occurs when:
        - Score improvement is less than threshold, OR
        - Score is already at 0 (perfect fit)

        Args:
            score_history: List of scores from successive trials
            threshold: Minimum improvement to continue (default: 0.5)

        Returns:
            bool: True if converged, False if should continue
        """
        if len(score_history) < 2:
            return False

        current = score_history[-1]
        previous = score_history[-2]

        # Perfect fit
        if current == 0:
            return True

        # Improvement less than threshold
        improvement = previous - current
        return improvement < threshold

    def print_analysis(self, biomech, score):
        """
        Pretty-print biomechanics analysis to console.

        Args:
            biomech: Dict from analyze_weight_bearing_frames()
            score: Float from compute_fit_score()
        """
        if biomech is None:
            print("⚠️  Insufficient data for analysis")
            return

        print(f"\n📊 BIOMECHANICS ANALYSIS")
        print(f"   Frames analyzed: {biomech['n_frames']} (weight-bearing only)")
        print(f"\n   Right Elbow: {biomech['avg_elbow_r']:.1f}° "
              f"(target: {self.target_elbow_ideal:.1f}°)")
        print(f"   Left Elbow:  {biomech['avg_elbow_l']:.1f}° "
              f"(target: {self.target_elbow_ideal:.1f}°)")
        print(f"   Average:     {biomech['avg_elbow']:.1f}° "
              f"(range: {self.target_elbow_min}-{self.target_elbow_max}°)")
        print(f"\n   Trunk Lean:  {biomech['avg_trunk']:.1f}° "
              f"(target: < {self.target_trunk_max}°)")
        print(f"   Asymmetry:   {biomech['elbow_asymmetry']:.1f}° (should be < 10°)")
        print(f"\n   FIT SCORE: {score:.2f} (lower is better)")

    def print_suggestions(self, suggestions):
        """
        Pretty-print adjustment suggestions to console.

        Args:
            suggestions: List from recommend_adjustment()
        """
        if not suggestions:
            print(f"\n💡 No adjustments needed")
            return

        print(f"\n💡 RECOMMENDED ADJUSTMENTS:")

        for i, sug in enumerate(suggestions, 1):
            priority_emoji = {
                'high': '🔴',
                'medium': '🟡',
                'info': '✅'
            }.get(sug['priority'], '⚪')

            if sug['type'] == 'success':
                print(f"   {priority_emoji} {sug['reason']}")

            elif sug['type'] == 'warning':
                print(f"   {priority_emoji} WARNING: {sug['reason']}")

            else:
                action = "RAISE" if sug['adjustment'] == 'increase' else "LOWER"
                part = "Handle" if sug['type'] == 'grip_height' else "Overall Crutch"
                inches = sug['amount_cm'] / 2.54
                print(f"   {priority_emoji} {i}. {action} {part} by "
                      f"{inches:.0f} inch (~{sug['amount_cm']:.1f}cm)")
                print(f"      Reason: {sug['reason']}")


# Quick test when run directly
if __name__ == '__main__':
    print("="*60)
    print("LIVE FIT OPTIMIZER - TEST")
    print("="*60)

    optimizer = LiveFitOptimizer()

    # Simulate different biomechanics scenarios
    test_cases = [
        {
            'name': 'Good fit',
            'avg_elbow_r': 157,
            'avg_elbow_l': 155,
            'avg_trunk': 8,
        },
        {
            'name': 'Elbows too straight + forward lean',
            'avg_elbow_r': 175,
            'avg_elbow_l': 172,
            'avg_trunk': 18,
        },
        {
            'name': 'Elbows too bent',
            'avg_elbow_r': 138,
            'avg_elbow_l': 140,
            'avg_trunk': 6,
        },
        {
            'name': 'Large asymmetry',
            'avg_elbow_r': 165,
            'avg_elbow_l': 145,
            'avg_trunk': 10,
        },
    ]

    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST: {case['name']}")
        print('='*60)

        # Build mock biomech dict
        biomech = {
            'avg_elbow_r': case['avg_elbow_r'],
            'avg_elbow_l': case['avg_elbow_l'],
            'avg_elbow': (case['avg_elbow_r'] + case['avg_elbow_l']) / 2,
            'avg_trunk': case['avg_trunk'],
            'std_elbow_r': 3.0,
            'std_elbow_l': 3.0,
            'n_frames': 50,
            'elbow_asymmetry': abs(case['avg_elbow_r'] - case['avg_elbow_l'])
        }

        score = optimizer.compute_fit_score(biomech)
        optimizer.print_analysis(biomech, score)

        mock_config = {
            'grip_height_cm': 85.0,
            'grip_setting_idx': 4,
            'overall_length_cm': 125.0,
            'overall_setting_idx': 3
        }

        suggestions = optimizer.recommend_adjustment(biomech, mock_config)
        optimizer.print_suggestions(suggestions)
