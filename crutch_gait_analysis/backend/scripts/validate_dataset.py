#!/usr/bin/env python3
"""Validate synthetic dataset quality"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict

def load_trials(filepath):
    """Load JSONL file into list of dicts"""
    trials = []
    with open(filepath, 'r') as f:
        for line in f:
            trials.append(json.loads(line))
    return trials

def validate_dataset(trials):
    """Run comprehensive validation checks"""

    print("🔍 DATASET VALIDATION\n")
    print("="*60)

    # Convert to DataFrame for analysis
    df = pd.DataFrame(trials)

    # CHECK 1: Basic structure
    print("\n1️⃣ BASIC STRUCTURE")
    print(f"   Total trials: {len(df)}")
    print(f"   Unique sessions: {df['session_id'].nunique()}")
    print(f"   Expected sessions: {len(df) // 3}")

    if len(df) % 3 == 0:
        print("   ✅ Trial count is multiple of 3")
    else:
        print(f"   ❌ Trial count ({len(df)}) is not multiple of 3")

    # CHECK 2: Required fields
    print("\n2️⃣ REQUIRED FIELDS")
    required = ['trial_id', 'session_id', 'user_height_cm', 'grip_height_cm',
                'overall_length_cm', 'score_total', 'is_best']

    for field in required:
        if field in df.columns:
            missing = df[field].isna().sum()
            if missing == 0:
                print(f"   ✅ {field}: {missing} missing")
            else:
                print(f"   ❌ {field}: {missing} missing")
        else:
            print(f"   ❌ {field}: COLUMN MISSING")

    # CHECK 3: Value ranges
    print("\n3️⃣ VALUE RANGES")

    checks = [
        ('Height', 'user_height_cm', 140, 210),
        ('Grip', 'grip_height_cm', 70, 100),
        ('Overall', 'overall_length_cm', 110, 145),
        ('Score', 'score_total', 0, 20)
    ]

    for name, col, min_val, max_val in checks:
        if col in df.columns:
            actual_min = df[col].min()
            actual_max = df[col].max()

            if min_val <= actual_min and actual_max <= max_val:
                print(f"   ✅ {name}: {actual_min:.1f} - {actual_max:.1f}")
            else:
                print(f"   ⚠️  {name}: {actual_min:.1f} - {actual_max:.1f} (expected {min_val}-{max_val})")

    # CHECK 4: Trial sequences
    print("\n4️⃣ TRIAL SEQUENCES")

    by_session = df.groupby('session_id')

    sessions_with_3 = sum(1 for _, g in by_session if len(g) == 3)
    sessions_with_1_best = sum(1 for _, g in by_session if g['is_best'].sum() == 1)

    total_sessions = len(by_session)

    print(f"   Sessions with 3 trials: {sessions_with_3}/{total_sessions}")
    print(f"   Sessions with 1 best: {sessions_with_1_best}/{total_sessions}")

    if sessions_with_3 == total_sessions:
        print("   ✅ All sessions have 3 trials")
    else:
        print(f"   ❌ {total_sessions - sessions_with_3} sessions missing trials")

    if sessions_with_1_best == total_sessions:
        print("   ✅ All sessions have exactly 1 best trial")
    else:
        print(f"   ❌ {total_sessions - sessions_with_1_best} sessions with incorrect best count")

    # CHECK 5: Score improvement
    print("\n5️⃣ SCORE IMPROVEMENT")

    trial1_scores = df[df['trial_index'] == 1]['score_total']
    trial3_scores = df[df['trial_index'] == 3]['score_total']
    best_scores = df[df['is_best'] == True]['score_total']

    print(f"   Trial 1 avg score: {trial1_scores.mean():.2f}")
    print(f"   Trial 3 avg score: {trial3_scores.mean():.2f}")
    print(f"   Best trials avg score: {best_scores.mean():.2f}")
    print(f"   Improvement: {trial1_scores.mean() - best_scores.mean():.2f}")

    if trial1_scores.mean() > best_scores.mean():
        print("   ✅ Scores improve from trial 1 to best")
    else:
        print("   ❌ Scores don't show improvement")

    # CHECK 6: Height distribution
    print("\n6️⃣ HEIGHT DISTRIBUTION")

    unique_heights = df.groupby('session_id')['user_height_cm'].first()

    print(f"   Mean: {unique_heights.mean():.1f} cm (expected ~170cm)")
    print(f"   Std: {unique_heights.std():.1f} cm (expected ~10cm)")
    print(f"   Range: {unique_heights.min():.1f} - {unique_heights.max():.1f} cm")

    if 165 < unique_heights.mean() < 175:
        print("   ✅ Height distribution looks realistic")
    else:
        print("   ⚠️  Height distribution may be skewed")

    # CHECK 7: Correlations
    print("\n7️⃣ CORRELATIONS")

    # Height vs Grip (should be positive)
    corr_height_grip = df.groupby('session_id').first()[['user_height_cm', 'grip_height_cm']].corr().iloc[0, 1]
    print(f"   Height ↔ Grip: {corr_height_grip:.3f} (expect > 0.7)")

    if corr_height_grip > 0.7:
        print("   ✅ Settings correlate with height")
    else:
        print("   ⚠️  Weak correlation")

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    print(f"\n✅ Dataset is {'READY' if sessions_with_3 == total_sessions else 'NOT READY'} for training")

    return df

if __name__ == '__main__':
    trials = load_trials('dataset/synthetic_trials.jsonl')
    df = validate_dataset(trials)
