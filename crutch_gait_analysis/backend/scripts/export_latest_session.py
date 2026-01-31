#!/usr/bin/env python3
"""
Export Latest Session Script
Finds the most recent session and exports trial outcomes for analysis.

Usage:
    python scripts/export_latest_session.py
    python scripts/export_latest_session.py --user john
    python scripts/export_latest_session.py --all
"""

import argparse
import glob
import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.trial_exporter import TrialOutcomeExporter, export_all_sessions


def find_latest_session(sessions_dir: str = 'sessions', user_id: str = None) -> str:
    """
    Find the most recent session file.

    Args:
        sessions_dir: Base sessions directory
        user_id: Optional user folder to search in

    Returns:
        Path to the most recent session file, or None if not found
    """
    session_files = []

    if user_id:
        # Search in specific user folder
        user_dir = os.path.join(sessions_dir, user_id)
        pattern = os.path.join(user_dir, 'session_*.json')
        session_files = glob.glob(pattern)
    else:
        # Search in root
        root_files = glob.glob(os.path.join(sessions_dir, 'session_*.json'))
        session_files.extend(root_files)

        # Search in all user subfolders
        subfolder_pattern = os.path.join(sessions_dir, '*', 'session_*.json')
        subfolder_files = glob.glob(subfolder_pattern)
        session_files.extend(subfolder_files)

    if not session_files:
        return None

    # Sort by modification time (most recent first)
    session_files.sort(key=os.path.getmtime, reverse=True)

    return session_files[0]


def export_latest(sessions_dir: str = 'sessions', output_dir: str = 'dataset',
                  user_id: str = None) -> None:
    """
    Export trials from the most recent session.

    Args:
        sessions_dir: Directory containing session files
        output_dir: Directory to save exported trials
        user_id: Optional user folder to search in
    """
    # Find latest session
    latest_session = find_latest_session(sessions_dir, user_id)

    if not latest_session:
        if user_id:
            print(f"No session files found for user '{user_id}' in {sessions_dir}/")
        else:
            print(f"No session files found in {sessions_dir}/")
        return

    print(f"\n📂 Exporting trials from: {latest_session}")

    # Load session to check for user-marked best
    with open(latest_session, 'r') as f:
        session_data = json.load(f)

    user_marked_best_idx = session_data['metadata'].get('best_trial_index')

    # Export trials
    try:
        exporter = TrialOutcomeExporter(latest_session)
        outcomes = exporter.export_trials(output_dir)
    except Exception as e:
        print(f"Error exporting trials: {e}")
        return

    if not outcomes:
        print("\nNo valid trials to export (need at least 10 weight-bearing frames per trial)")
        return

    # Find best trial by score (lowest is best)
    best_by_score = min(outcomes, key=lambda x: x['score_total'])

    # Find user-marked best if any
    user_marked_best = None
    for outcome in outcomes:
        if outcome.get('is_best'):
            user_marked_best = outcome
            break

    # Print recommendation
    print(f"\n{'='*50}")
    print("⭐ BEST TRIAL (lowest score):")
    print(f"{'='*50}")
    print(f"   Trial: {best_by_score['trial_id']}")
    print(f"   Grip: Hole #{best_by_score['grip_setting_idx'] + 1} "
          f"({best_by_score['grip_height_cm']:.1f}cm / "
          f"{best_by_score['grip_height_cm']/2.54:.1f}\")")
    print(f"   Overall: Position #{best_by_score['overall_setting_idx'] + 1} "
          f"({best_by_score['overall_length_cm']:.1f}cm / "
          f"{best_by_score['overall_length_cm']/2.54:.1f}\")")
    print(f"   Score: {best_by_score['score_total']}")

    if best_by_score['top_issues']:
        print(f"   Top issues: {', '.join(best_by_score['top_issues'])}")

    # Show user-marked best if different
    if user_marked_best and user_marked_best['trial_id'] != best_by_score['trial_id']:
        print(f"\n✨ User marked {user_marked_best['trial_id']} as best subjectively")
        print(f"   (Score: {user_marked_best['score_total']})")

    # Quick comparison table
    if len(outcomes) > 1:
        print(f"\n{'='*50}")
        print("TRIAL COMPARISON:")
        print(f"{'='*50}")
        print(f"{'Trial':<8} {'Score':<8} {'Overall':<12} {'Grip':<12} {'Best':<6}")
        print("-" * 50)

        for outcome in sorted(outcomes, key=lambda x: x['trial_index']):
            trial_id = outcome['trial_id']
            score = outcome['score_total']
            overall = f"#{outcome['overall_setting_idx'] + 1}"
            grip = f"#{outcome['grip_setting_idx'] + 1}"

            markers = []
            if outcome['trial_id'] == best_by_score['trial_id']:
                markers.append("📊")
            if outcome.get('is_best'):
                markers.append("⭐")

            best_str = " ".join(markers) if markers else ""

            print(f"{trial_id:<8} {score:<8.1f} {overall:<12} {grip:<12} {best_str}")

        print("-" * 50)
        print("📊 = Best by score, ⭐ = User marked as best")


def main():
    parser = argparse.ArgumentParser(
        description='Export trial outcomes from the latest session'
    )
    parser.add_argument(
        '--user', '-u',
        type=str,
        help='Export only sessions from this user folder'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Export all sessions (not just the latest)'
    )
    parser.add_argument(
        '--sessions-dir', '-s',
        type=str,
        default='sessions',
        help='Sessions directory (default: sessions)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='dataset',
        help='Output directory for trials.jsonl (default: dataset)'
    )
    parser.add_argument(
        '--exclude', '-e',
        type=str,
        nargs='+',
        help='User folders to exclude (e.g., --exclude test_run calibration)'
    )

    args = parser.parse_args()

    print("="*60)
    print("TRIAL OUTCOME EXPORTER")
    print("="*60)

    if args.all:
        # Export all sessions
        export_all_sessions(
            args.sessions_dir,
            args.output_dir,
            user_filter=args.user,
            exclude_users=args.exclude
        )
    else:
        # Export only latest session
        export_latest(args.sessions_dir, args.output_dir, args.user)


if __name__ == '__main__':
    main()
