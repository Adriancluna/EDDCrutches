#!/usr/bin/env python3
"""
Interactive crutch fitting session with AI recommendations.

This script provides a complete fitting workflow:
1. User enters measurements
2. AI recommends initial settings
3. User walks with crutches
4. System analyzes biomechanics
5. Optimizer suggests adjustments
6. Repeat until optimal fit

Usage:
    python scripts/interactive_fitting_session.py
"""

import os
import sys
import glob
import json
from recommender.knn_recommender import KNNRecommender
from recommender.live_optimizer import LiveFitOptimizer
from config.crutch_models import DEFAULT_CRUTCH


def load_latest_session():
    """Load most recent session file"""
    session_files = glob.glob('sessions/session_*.json')
    if not session_files:
        return None

    latest = max(session_files, key=os.path.getmtime)

    with open(latest, 'r') as f:
        return json.load(f)


def main():
    print("="*70)
    print("INTERACTIVE CRUTCH FITTING SESSION")
    print("="*70)

    # Load recommender
    model_path = 'models/knn_recommender.pkl'
    if not os.path.exists(model_path):
        print(f"\n❌ Error: Trained model not found at {model_path}")
        print("   Run: python scripts/train_recommender.py")
        return 1

    print(f"\n📥 Loading AI recommender...")
    recommender = KNNRecommender()
    recommender.load(model_path)

    print(f"\n📥 Loading live optimizer...")
    optimizer = LiveFitOptimizer()

    # Get user info
    print(f"\n" + "="*70)
    print("STEP 1: USER MEASUREMENTS")
    print("="*70)

    try:
        height = float(input("\nEnter your height in cm: "))
        weight_input = input("Enter your weight in kg (press Enter to skip): ")
        weight = float(weight_input) if weight_input.strip() else None
    except ValueError:
        print("❌ Invalid input")
        return 1

    # Get initial recommendation
    print(f"\n" + "="*70)
    print("STEP 2: AI INITIAL RECOMMENDATION")
    print("="*70)

    recommendation = recommender.predict(height, weight)
    recommender.print_recommendation(recommendation)

    # Current config
    current_config = {
        'grip_height_cm': recommendation['grip_height_cm'],
        'grip_setting_idx': recommendation['grip_setting_idx'],
        'overall_length_cm': recommendation['overall_length_cm'],
        'overall_setting_idx': recommendation['overall_setting_idx']
    }

    # Interactive loop
    iteration = 1
    score_history = []

    while True:
        print(f"\n" + "="*70)
        print(f"ITERATION {iteration}")
        print("="*70)

        if iteration == 1:
            print(f"\n📋 Set your crutches to the recommended settings above")
        else:
            print(f"\n📋 Current settings:")
            print(f"   Grip: Hole #{current_config['grip_setting_idx'] + 1} ({current_config['grip_height_cm']:.1f}cm)")
            print(f"   Overall: Setting #{current_config['overall_setting_idx'] + 1} ({current_config['overall_length_cm']:.1f}cm)")

        print(f"\n▶️  Now walk with your crutches for 30 seconds")
        print(f"   (Run your main application to record)")

        input("\nPress Enter when recording is complete...")

        # Load latest session
        session_data = load_latest_session()
        if session_data is None:
            print("❌ No session data found. Please record a session first.")
            return 1

        # Analyze biomechanics
        print(f"\n🔍 Analyzing your gait...")
        biomech = optimizer.analyze_weight_bearing_frames(session_data['frames'])

        if biomech is None:
            print("❌ Insufficient data. Please walk longer (at least 30 seconds).")
            continue

        score = optimizer.compute_fit_score(biomech)
        score_history.append(score)

        optimizer.print_analysis(biomech, score)

        # Check for convergence
        if len(score_history) > 1:
            improvement = score_history[-2] - score_history[-1]
            print(f"\n   Improvement from last trial: {improvement:+.2f}")

            if improvement < 0.5 and score < 2.0:
                print(f"\n🎉 FIT CONVERGED!")
                print(f"   Your crutches are optimally adjusted.")
                break

            if improvement < 0:
                print(f"\n⚠️  Score got worse. Previous settings may have been better.")

        # Get suggestions
        suggestions = optimizer.recommend_adjustment(biomech, current_config)
        optimizer.print_suggestions(suggestions)

        # Check if done
        success = any(s['type'] == 'success' for s in suggestions)
        if success:
            print(f"\n🎉 OPTIMAL FIT ACHIEVED!")
            break

        # Ask to continue
        print(f"\n" + "-"*70)
        proceed = input("Apply adjustment and test again? (y/n): ").lower()

        if proceed != 'y':
            print("\n🛑 Fitting session ended")
            break

        # Apply first non-warning suggestion
        for sug in suggestions:
            if sug['type'] not in ['success', 'warning']:
                new_config = optimizer.apply_suggestion(sug, current_config)

                print(f"\n✏️  NEW SETTINGS:")
                print(f"   Grip: Hole #{new_config['grip_setting_idx'] + 1} ({new_config['grip_height_cm']:.1f}cm)")
                print(f"   Overall: Setting #{new_config['overall_setting_idx'] + 1} ({new_config['overall_length_cm']:.1f}cm)")

                current_config = new_config
                break

        iteration += 1

        if iteration > 5:
            print("\n⚠️  Maximum iterations reached. You may need professional fitting.")
            break

    # Final summary
    print(f"\n" + "="*70)
    print("FITTING SESSION COMPLETE")
    print("="*70)

    if score_history:
        print(f"\n📊 Score History: {' → '.join(f'{s:.2f}' for s in score_history)}")
        if len(score_history) > 1:
            print(f"   Total improvement: {score_history[0] - score_history[-1]:.2f}")

    print(f"\n✅ FINAL SETTINGS:")
    print(f"   Grip: Hole #{current_config['grip_setting_idx'] + 1} ({current_config['grip_height_cm']:.1f}cm / {current_config['grip_height_cm']/2.54:.1f}\")")
    print(f"   Overall: Setting #{current_config['overall_setting_idx'] + 1} ({current_config['overall_length_cm']:.1f}cm / {current_config['overall_length_cm']/2.54:.1f}\")")

    return 0


if __name__ == '__main__':
    sys.exit(main())
