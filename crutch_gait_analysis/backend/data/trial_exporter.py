"""
Trial Outcome Exporter
Analyzes sessions by trial, exports structured dataset for ML
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np


class TrialOutcomeExporter:
    """
    Exports trial-level outcomes from session data.

    Analyzes each trial's biomechanics and exports structured data
    suitable for machine learning or statistical analysis.
    """

    # Penalty weights for evaluation statuses
    PENALTY_WEIGHTS = {
        'critical': 4.0,
        'warning': 2.0,
        'caution': 1.0,
        'good': 0.0
    }

    # Key biomechanics metrics to track
    KEY_METRICS = ['elbow_r', 'elbow_l', 'trunk_lean', 'knee_r', 'knee_l']

    def __init__(self, session_filepath: str):
        """
        Initialize exporter with a session file.

        Args:
            session_filepath: Path to session JSON file

        Raises:
            FileNotFoundError: If session file doesn't exist
            json.JSONDecodeError: If session file is invalid JSON
        """
        if not os.path.exists(session_filepath):
            raise FileNotFoundError(f"Session file not found: {session_filepath}")

        with open(session_filepath, 'r') as f:
            self.session_data = json.load(f)

        self.session_id = self.session_data['metadata']['session_id']
        self.frames = self.session_data['frames']
        self.user_height_cm = self.session_data['metadata'].get('user_height_cm')

    def split_by_trials(self) -> Dict[str, List[dict]]:
        """
        Group frames by trial_id.

        Returns:
            dict mapping trial_id to list of frames
            e.g., {'T001': [frame1, frame2, ...], 'T002': [...]}
        """
        trials = defaultdict(list)

        for frame in self.frames:
            trial_id = frame['measurements'].get('trial_id', 'T001')
            trials[trial_id].append(frame)

        return dict(trials)

    def compute_trial_outcome(self, trial_frames: List[dict]) -> Optional[Dict]:
        """
        Compute outcome metrics for one trial.

        Filters to weight-bearing frames only (most clinically relevant).
        Computes penalty scores, metric statistics, and issue frequency.

        Args:
            trial_frames: List of frame dicts for this trial

        Returns:
            dict with trial config and computed outcomes, or None if insufficient data
        """
        if len(trial_frames) == 0:
            return None

        # Filter to weight-bearing frames (most important for fit analysis)
        wb_frames = [
            f for f in trial_frames
            if f['measurements'].get('gait_phase', '').startswith('WEIGHT_BEARING')
        ]

        if len(wb_frames) < 10:
            return None  # Not enough weight-bearing data

        # ========================================
        # Compute penalty score
        # ========================================
        total_penalty = 0.0
        for frame in wb_frames:
            evals = frame.get('evaluations', {})
            for metric, eval_data in evals.items():
                status = eval_data.get('status', 'good')
                total_penalty += self.PENALTY_WEIGHTS.get(status, 0.0)

        score_total = total_penalty / len(wb_frames)

        # ========================================
        # Mean/std for key metrics
        # ========================================
        metrics_mean = {}
        metrics_std = {}

        for metric in self.KEY_METRICS:
            values = []
            for f in wb_frames:
                val = f['measurements'].get(metric)
                if val is not None:
                    values.append(val)

            if values:
                metrics_mean[metric] = float(np.mean(values))
                metrics_std[metric] = float(np.std(values))
            else:
                metrics_mean[metric] = None
                metrics_std[metric] = None

        # ========================================
        # Top persistent issues
        # ========================================
        issue_counts = defaultdict(int)
        for frame in wb_frames:
            for issue in frame.get('persistent_issues', []):
                issue_counts[issue] += 1

        top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_issues = [issue for issue, count in top_issues]

        # ========================================
        # Phase distribution (all frames)
        # ========================================
        phase_counts = defaultdict(int)
        for frame in trial_frames:
            phase = frame['measurements'].get('gait_phase', 'UNKNOWN')
            phase_counts[phase] += 1

        total_frames = len(trial_frames)
        phase_distribution = {
            phase: count / total_frames
            for phase, count in phase_counts.items()
        }

        # ========================================
        # Extract trial config from first frame
        # ========================================
        first_frame = trial_frames[0]['measurements']
        last_frame = trial_frames[-1]

        return {
            # Trial identification
            'trial_id': first_frame.get('trial_id', 'UNKNOWN'),
            'trial_index': first_frame.get('trial_index', 0),
            'session_id': self.session_id,

            # Crutch configuration
            'crutch_model_id': first_frame.get('crutch_model_id', 'unknown'),
            'overall_length_cm': first_frame.get('overall_length_cm', 0),
            'grip_height_cm': first_frame.get('grip_height_cm', 0),
            'overall_setting_idx': first_frame.get('overall_setting_idx', 0),
            'grip_setting_idx': first_frame.get('grip_setting_idx', 0),

            # User context
            'user_height_cm': self.user_height_cm,

            # Timing
            'start_time_s': round(trial_frames[0]['timestamp'], 2),
            'end_time_s': round(last_frame['timestamp'], 2),
            'duration_s': round(last_frame['timestamp'] - trial_frames[0]['timestamp'], 2),

            # Frame counts
            'num_frames': len(trial_frames),
            'num_wb_frames': len(wb_frames),
            'wb_frame_ratio': round(len(wb_frames) / len(trial_frames), 3),

            # Outcome metrics
            'score_total': round(score_total, 2),
            'metrics_mean': {k: round(v, 1) if v else None for k, v in metrics_mean.items()},
            'metrics_std': {k: round(v, 2) if v else None for k, v in metrics_std.items()},
            'top_issues': top_issues,
            'phase_distribution': {k: round(v, 3) for k, v in phase_distribution.items()},

            # User feedback (populated if available)
            'user_rating': None,
            'user_issues': [],
            'is_best': False
        }

    def export_trials(self, output_dir: str = 'dataset') -> List[Dict]:
        """
        Export trial outcomes to JSONL file.

        Splits session into trials, computes outcomes, and appends
        to a JSON Lines file (one JSON object per line).

        Args:
            output_dir: Directory to save output (created if needed)

        Returns:
            List of outcome dicts that were exported
        """
        os.makedirs(output_dir, exist_ok=True)

        trials_dict = self.split_by_trials()
        outcomes = []

        print(f"\n📊 Analyzing {len(trials_dict)} trial(s) from session {self.session_id}...")

        for trial_id in sorted(trials_dict.keys()):
            frames = trials_dict[trial_id]
            outcome = self.compute_trial_outcome(frames)

            if outcome:
                # Check if this trial was marked as best
                best_trial_idx = self.session_data['metadata'].get('best_trial_index')
                if best_trial_idx and outcome['trial_index'] == best_trial_idx:
                    outcome['is_best'] = True

                outcomes.append(outcome)
                best_marker = " ⭐ BEST" if outcome['is_best'] else ""
                print(f"  {trial_id}: score={outcome['score_total']:.1f}, "
                      f"frames={outcome['num_frames']}{best_marker}")
            else:
                print(f"  {trial_id}: ⚠️  Skipped (insufficient weight-bearing frames)")

        if not outcomes:
            print("\n⚠️  No trials with sufficient data to export")
            return []

        # Append to JSONL file
        trials_file = os.path.join(output_dir, 'trials.jsonl')
        with open(trials_file, 'a') as f:
            for outcome in outcomes:
                f.write(json.dumps(outcome) + '\n')

        print(f"\n✅ Exported {len(outcomes)} trial(s) to {trials_file}")

        return outcomes

    def get_trial_summary(self) -> Dict:
        """
        Get a quick summary of trials in this session.

        Returns:
            dict with trial count, best trial, score range
        """
        trials_dict = self.split_by_trials()
        outcomes = []

        for trial_id, frames in trials_dict.items():
            outcome = self.compute_trial_outcome(frames)
            if outcome:
                outcomes.append(outcome)

        if not outcomes:
            return {'trial_count': 0, 'valid_trials': 0}

        scores = [o['score_total'] for o in outcomes]

        return {
            'trial_count': len(trials_dict),
            'valid_trials': len(outcomes),
            'best_score': min(scores),
            'worst_score': max(scores),
            'best_trial_id': outcomes[scores.index(min(scores))]['trial_id'],
            'outcomes': outcomes
        }


def export_all_sessions(sessions_dir: str = 'sessions', output_dir: str = 'dataset',
                        user_filter: str = None, exclude_users: List[str] = None) -> int:
    """
    Export trials from all session files in a directory (including user subfolders).

    Args:
        sessions_dir: Directory containing session JSON files (with user subfolders)
        output_dir: Directory to save output
        user_filter: If specified, only export sessions from this user folder
        exclude_users: List of user folders to skip (e.g., ['test_run', 'calibration'])

    Returns:
        Total number of trials exported

    Example folder structure:
        sessions/
            john/
                session_20260128_143022.json
                session_20260128_150000.json
            test_run/
                session_20260128_120000.json
            anonymous/
                session_20260128_100000.json
    """
    import glob

    exclude_users = exclude_users or []

    # Find all session files (both in root and in user subfolders)
    session_files = []

    # Check root directory
    root_files = glob.glob(os.path.join(sessions_dir, 'session_*.json'))
    session_files.extend(root_files)

    # Check user subfolders
    user_pattern = os.path.join(sessions_dir, '*', 'session_*.json')
    subfolder_files = glob.glob(user_pattern)
    session_files.extend(subfolder_files)

    if not session_files:
        print(f"No session files found in {sessions_dir}/")
        return 0

    # Filter by user if specified
    if user_filter:
        session_files = [f for f in session_files if f"/{user_filter}/" in f or f"\\{user_filter}\\" in f]

    # Exclude specific users
    for exclude_user in exclude_users:
        session_files = [f for f in session_files
                        if f"/{exclude_user}/" not in f and f"\\{exclude_user}\\" not in f]

    if not session_files:
        print(f"No session files found after filtering")
        return 0

    total_trials = 0
    users_processed = set()

    for filepath in sorted(session_files):
        # Extract user from path
        parts = filepath.replace('\\', '/').split('/')
        if len(parts) >= 3 and parts[-2] != sessions_dir.replace('\\', '/').split('/')[-1]:
            user = parts[-2]
        else:
            user = "root"
        users_processed.add(user)

        try:
            exporter = TrialOutcomeExporter(filepath)
            outcomes = exporter.export_trials(output_dir)
            total_trials += len(outcomes)
        except Exception as e:
            print(f"⚠️  Error processing {filepath}: {e}")

    print(f"\n{'='*50}")
    print(f"Total: {total_trials} trials exported from {len(session_files)} session(s)")
    print(f"Users: {', '.join(sorted(users_processed))}")

    return total_trials


def export_user_sessions(user_id: str, sessions_dir: str = 'sessions',
                         output_dir: str = 'dataset') -> int:
    """
    Export trials from a specific user's sessions only.

    Args:
        user_id: User folder name to export
        sessions_dir: Base sessions directory
        output_dir: Directory to save output

    Returns:
        Total number of trials exported
    """
    return export_all_sessions(sessions_dir, output_dir, user_filter=user_id)
