#!/usr/bin/env python3
"""
Synthetic Dataset Generator for Crutch Fitting ML Training

Generates synthetic crutch fitting trials from ANSUR II anthropometric data.
Each subject gets 3 trials simulating a typical fitting progression:
- Trial 1: Initial estimate (with typical error)
- Trial 2: First correction (improved)
- Trial 3: Final optimization (near-optimal)

This provides bootstrap training data before collecting real user sessions.

Usage:
    python dataset/synthetic_dataset_generator.py
    python dataset/synthetic_dataset_generator.py --n-subjects 5000
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.clinical_guidelines import (
    estimate_ideal_grip_height,
    estimate_ideal_overall_length,
    calculate_fit_penalty,
    BIOMECHANICS_TARGETS,
    ANTHROPOMETRIC_RATIOS
)
from config import DEFAULT_CRUTCH


class SyntheticDatasetGenerator:
    """
    Generates synthetic crutch fitting trial data from ANSUR anthropometric measurements.

    Simulates the typical fitting process where users start with an initial estimate,
    make adjustments based on feedback, and converge toward optimal settings.
    """

    def __init__(self, ansur_csv_path: str = 'dataset/ansur_simplified.csv'):
        """
        Initialize generator with ANSUR data.

        Args:
            ansur_csv_path: Path to preprocessed ANSUR CSV file

        Raises:
            FileNotFoundError: If ANSUR file doesn't exist
        """
        if not os.path.exists(ansur_csv_path):
            raise FileNotFoundError(
                f"ANSUR data not found at {ansur_csv_path}\n"
                "Run: python scripts/preprocess_ansur.py --auto"
            )

        self.ansur_data = pd.read_csv(ansur_csv_path)
        print(f"Loaded {len(self.ansur_data)} subjects from ANSUR dataset")

        # Device profile for settings conversion
        self.device = DEFAULT_CRUTCH

        # Set random seed for reproducibility
        np.random.seed(42)

    def estimate_biomechanics(self, grip_error_cm: float, overall_error_cm: float) -> Dict:
        """
        Simulate expected biomechanics based on crutch setting errors.

        Models the relationship between crutch fit and body mechanics:
        - Grip too high → elbow too straight (extended)
        - Grip too low → elbow too bent (flexed)
        - Overall too short → forward trunk lean (compensation)
        - Overall too tall → less stable, may hyperextend

        Args:
            grip_error_cm: Difference from ideal grip height (positive = too high)
            overall_error_cm: Difference from ideal overall length (positive = too tall)

        Returns:
            dict with 'elbow_angle' and 'trunk_lean' (degrees)
        """
        # Base ideal values from clinical guidelines
        ideal_elbow = BIOMECHANICS_TARGETS['elbow_angle_ideal']
        ideal_trunk = BIOMECHANICS_TARGETS['trunk_lean_ideal']

        # Grip error affects elbow angle
        # Too high grip → more elbow extension (straighter arm)
        # Sensitivity: approximately 3° change per cm of grip error
        elbow_effect = grip_error_cm * 3.0
        elbow_angle = ideal_elbow + elbow_effect

        # Overall length error affects trunk lean
        # Too short → lean forward to compensate
        # Too tall → may lean back or be unstable
        # Sensitivity: approximately 2° change per cm of length error
        trunk_effect = -overall_error_cm * 2.0
        trunk_lean = ideal_trunk + trunk_effect

        # Add realistic measurement noise
        # Elbow: ±2° variation (natural movement variance)
        elbow_noise = np.random.normal(0, 2.0)
        # Trunk: ±1.5° variation (posture variance)
        trunk_noise = np.random.normal(0, 1.5)

        elbow_angle += elbow_noise
        trunk_lean += trunk_noise

        # Clamp to physiologically realistic ranges
        elbow_angle = np.clip(elbow_angle, 130, 180)  # 130° (very bent) to 180° (straight)
        trunk_lean = np.clip(trunk_lean, 0, 25)  # 0° (upright) to 25° (severe lean)

        return {
            'elbow_angle': round(elbow_angle, 1),
            'trunk_lean': round(trunk_lean, 1)
        }

    def generate_trial_sequence(self, subject_row: pd.Series) -> List[Dict]:
        """
        Generate a sequence of 3 fitting trials for one subject.

        Simulates a typical fitting session:
        - Trial 1: Initial estimate based on height (typical ±3-5cm error)
        - Trial 2: After first feedback/adjustment (error reduced ~60%)
        - Trial 3: Final optimization (near-optimal, ±1cm error)

        Args:
            subject_row: Pandas Series with subject measurements

        Returns:
            List of 3 trial dicts with settings, biomechanics, and scores
        """
        height_cm = subject_row['height_cm']
        sex = subject_row.get('sex', 'M')

        # Use measured values if available, otherwise estimate
        wrist_cm = subject_row.get('wrist_height_cm')
        if pd.isna(wrist_cm) or wrist_cm is None:
            wrist_cm = height_cm * ANTHROPOMETRIC_RATIOS['wrist_to_height']

        axilla_cm = subject_row.get('axilla_height_cm')
        if pd.isna(axilla_cm) or axilla_cm is None:
            axilla_cm = height_cm * ANTHROPOMETRIC_RATIOS['axilla_to_height']

        # Calculate ideal settings
        ideal_grip_cm = estimate_ideal_grip_height(height_cm, wrist_cm)
        ideal_overall_cm = estimate_ideal_overall_length(height_cm, axilla_cm, method='axilla')

        trials = []

        # ========================================
        # Trial 1: Initial guess with typical error
        # ========================================
        # Real users typically start with ±3-5cm error
        # Slight bias toward too-tall (common mistake)
        grip_error_1 = np.random.normal(0.5, 3.5)  # Slight high bias
        overall_error_1 = np.random.normal(1.0, 3.5)  # Slight tall bias

        trial1_grip = ideal_grip_cm + grip_error_1
        trial1_overall = ideal_overall_cm + overall_error_1

        # Convert to device settings (clamped to valid range)
        grip_info_1 = self.device.cm_to_setting(trial1_grip, 'grip')
        overall_info_1 = self.device.cm_to_setting(trial1_overall, 'overall')

        # Recalculate actual error after clamping/rounding to device settings
        actual_grip_error_1 = grip_info_1['actual_cm'] - ideal_grip_cm
        actual_overall_error_1 = overall_info_1['actual_cm'] - ideal_overall_cm

        # Simulate biomechanics
        biomech_1 = self.estimate_biomechanics(actual_grip_error_1, actual_overall_error_1)
        score_1 = calculate_fit_penalty(biomech_1['elbow_angle'], biomech_1['trunk_lean'])

        trials.append({
            'trial_index': 1,
            'grip_height_cm': grip_info_1['actual_cm'],
            'grip_setting_idx': grip_info_1['setting_idx'],
            'overall_length_cm': overall_info_1['actual_cm'],
            'overall_setting_idx': overall_info_1['setting_idx'],
            'ideal_grip_cm': round(ideal_grip_cm, 1),
            'ideal_overall_cm': round(ideal_overall_cm, 1),
            'grip_error_cm': round(actual_grip_error_1, 1),
            'overall_error_cm': round(actual_overall_error_1, 1),
            'simulated_elbow': biomech_1['elbow_angle'],
            'simulated_trunk': biomech_1['trunk_lean'],
            'score_total': round(score_1, 2),
            'is_best': False
        })

        # ========================================
        # Trial 2: Partial correction
        # ========================================
        # After first feedback, error typically reduced by 50-70%
        correction_factor = np.random.uniform(0.3, 0.5)  # Keep 30-50% of error
        grip_error_2 = actual_grip_error_1 * correction_factor
        overall_error_2 = actual_overall_error_1 * correction_factor

        trial2_grip = ideal_grip_cm + grip_error_2
        trial2_overall = ideal_overall_cm + overall_error_2

        grip_info_2 = self.device.cm_to_setting(trial2_grip, 'grip')
        overall_info_2 = self.device.cm_to_setting(trial2_overall, 'overall')

        actual_grip_error_2 = grip_info_2['actual_cm'] - ideal_grip_cm
        actual_overall_error_2 = overall_info_2['actual_cm'] - ideal_overall_cm

        biomech_2 = self.estimate_biomechanics(actual_grip_error_2, actual_overall_error_2)
        score_2 = calculate_fit_penalty(biomech_2['elbow_angle'], biomech_2['trunk_lean'])

        trials.append({
            'trial_index': 2,
            'grip_height_cm': grip_info_2['actual_cm'],
            'grip_setting_idx': grip_info_2['setting_idx'],
            'overall_length_cm': overall_info_2['actual_cm'],
            'overall_setting_idx': overall_info_2['setting_idx'],
            'ideal_grip_cm': round(ideal_grip_cm, 1),
            'ideal_overall_cm': round(ideal_overall_cm, 1),
            'grip_error_cm': round(actual_grip_error_2, 1),
            'overall_error_cm': round(actual_overall_error_2, 1),
            'simulated_elbow': biomech_2['elbow_angle'],
            'simulated_trunk': biomech_2['trunk_lean'],
            'score_total': round(score_2, 2),
            'is_best': False
        })

        # ========================================
        # Trial 3: Near-optimal (final adjustment)
        # ========================================
        # Final trial has small random error (±1cm)
        grip_error_3 = np.random.normal(0, 0.8)
        overall_error_3 = np.random.normal(0, 0.8)

        trial3_grip = ideal_grip_cm + grip_error_3
        trial3_overall = ideal_overall_cm + overall_error_3

        grip_info_3 = self.device.cm_to_setting(trial3_grip, 'grip')
        overall_info_3 = self.device.cm_to_setting(trial3_overall, 'overall')

        actual_grip_error_3 = grip_info_3['actual_cm'] - ideal_grip_cm
        actual_overall_error_3 = overall_info_3['actual_cm'] - ideal_overall_cm

        biomech_3 = self.estimate_biomechanics(actual_grip_error_3, actual_overall_error_3)
        score_3 = calculate_fit_penalty(biomech_3['elbow_angle'], biomech_3['trunk_lean'])

        trials.append({
            'trial_index': 3,
            'grip_height_cm': grip_info_3['actual_cm'],
            'grip_setting_idx': grip_info_3['setting_idx'],
            'overall_length_cm': overall_info_3['actual_cm'],
            'overall_setting_idx': overall_info_3['setting_idx'],
            'ideal_grip_cm': round(ideal_grip_cm, 1),
            'ideal_overall_cm': round(ideal_overall_cm, 1),
            'grip_error_cm': round(actual_grip_error_3, 1),
            'overall_error_cm': round(actual_overall_error_3, 1),
            'simulated_elbow': biomech_3['elbow_angle'],
            'simulated_trunk': biomech_3['trunk_lean'],
            'score_total': round(score_3, 2),
            'is_best': False
        })

        # Mark best trial (lowest score)
        scores = [t['score_total'] for t in trials]
        best_idx = np.argmin(scores)
        trials[best_idx]['is_best'] = True

        return trials

    def generate_dataset(self, n_subjects: int = 1000,
                         output_path: str = 'dataset/synthetic_trials.jsonl',
                         random_seed: int = 42) -> List[Dict]:
        """
        Generate complete synthetic dataset.

        Args:
            n_subjects: Number of ANSUR subjects to sample
            output_path: Where to save JSONL file
            random_seed: Random seed for reproducibility

        Returns:
            List of all generated trial dicts
        """
        np.random.seed(random_seed)

        # Sample from ANSUR (with replacement if requesting more than available)
        if n_subjects > len(self.ansur_data):
            subjects = self.ansur_data.sample(n=n_subjects, replace=True, random_state=random_seed)
            print(f"Note: Sampling with replacement (requested {n_subjects}, have {len(self.ansur_data)})")
        else:
            subjects = self.ansur_data.sample(n=n_subjects, random_state=random_seed)

        print(f"\n{'='*60}")
        print("SYNTHETIC DATASET GENERATION")
        print(f"{'='*60}")
        print(f"\nSubjects: {n_subjects}")
        print(f"Trials per subject: 3")
        print(f"Total trials: {n_subjects * 3}")
        print(f"Output: {output_path}\n")

        all_trials = []
        base_date = datetime(2025, 1, 1)

        for idx, (_, subject) in enumerate(subjects.iterrows()):
            # Generate unique IDs
            user_id = f"SYN_{idx+1:05d}"
            session_date = base_date + pd.Timedelta(days=idx // 100)
            session_id = f"{session_date.strftime('%Y%m%d')}_{user_id}"

            # Generate 3 trials for this subject
            trials = self.generate_trial_sequence(subject)

            # Add metadata to each trial
            for trial in trials:
                trial.update({
                    'trial_id': f"T{trial['trial_index']:03d}",
                    'session_id': session_id,
                    'user_id': user_id,
                    'user_height_cm': round(float(subject['height_cm']), 1),
                    'user_weight_kg': round(float(subject.get('weight_kg', 70)), 1),
                    'user_sex': subject.get('sex', 'U'),
                    'crutch_model_id': self.device.model_id,
                    'data_source': 'synthetic_ansur',
                    # Simulated frame counts (typical 30fps for 30 seconds)
                    'num_frames': 900,
                    'num_wb_frames': int(np.random.uniform(400, 500)),
                    'duration_s': 30.0
                })
                all_trials.append(trial)

            # Progress indicator
            if (idx + 1) % 500 == 0:
                print(f"   Generated {idx + 1}/{n_subjects} subjects...")

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        # Save to JSONL
        with open(output_path, 'w') as f:
            for trial in all_trials:
                f.write(json.dumps(trial) + '\n')

        # Calculate statistics
        self._print_statistics(all_trials, output_path)

        return all_trials

    def _print_statistics(self, all_trials: List[Dict], output_path: str) -> None:
        """Print dataset statistics."""
        best_trials = [t for t in all_trials if t['is_best']]
        trial1_scores = [t['score_total'] for t in all_trials if t['trial_index'] == 1]
        trial2_scores = [t['score_total'] for t in all_trials if t['trial_index'] == 2]
        trial3_scores = [t['score_total'] for t in all_trials if t['trial_index'] == 3]
        best_scores = [t['score_total'] for t in best_trials]

        print(f"\n{'='*60}")
        print("GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"\nTotal trials: {len(all_trials)}")
        print(f"Saved to: {output_path}")

        print(f"\n📊 Score Statistics:")
        print(f"   Trial 1 (initial):     avg={np.mean(trial1_scores):.2f}, std={np.std(trial1_scores):.2f}")
        print(f"   Trial 2 (adjusted):    avg={np.mean(trial2_scores):.2f}, std={np.std(trial2_scores):.2f}")
        print(f"   Trial 3 (optimized):   avg={np.mean(trial3_scores):.2f}, std={np.std(trial3_scores):.2f}")
        print(f"   Best trials:           avg={np.mean(best_scores):.2f}, std={np.std(best_scores):.2f}")

        improvement = np.mean(trial1_scores) - np.mean(best_scores)
        print(f"\n📈 Average improvement: {improvement:.2f} points ({improvement/np.mean(trial1_scores)*100:.1f}%)")

        # Which trial was best most often
        best_trial_counts = {1: 0, 2: 0, 3: 0}
        for t in best_trials:
            best_trial_counts[t['trial_index']] += 1

        print(f"\n🏆 Best trial distribution:")
        for trial_idx, count in sorted(best_trial_counts.items()):
            pct = count / len(best_trials) * 100
            print(f"   Trial {trial_idx}: {count} ({pct:.1f}%)")

        # Height distribution
        heights = [t['user_height_cm'] for t in all_trials if t['trial_index'] == 1]
        print(f"\n📏 Height range: {min(heights):.1f} - {max(heights):.1f} cm")
        print(f"   Mean: {np.mean(heights):.1f} cm")


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic crutch fitting trials from ANSUR data'
    )
    parser.add_argument(
        '--n-subjects', '-n',
        type=int,
        default=1000,
        help='Number of subjects to generate (default: 1000)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='dataset/synthetic_trials.jsonl',
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--ansur-path', '-a',
        type=str,
        default='dataset/ansur_simplified.csv',
        help='Path to preprocessed ANSUR CSV'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    try:
        generator = SyntheticDatasetGenerator(args.ansur_path)
        generator.generate_dataset(
            n_subjects=args.n_subjects,
            output_path=args.output,
            random_seed=args.seed
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
