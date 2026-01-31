"""
Session data collection
"""
import json
from datetime import datetime
import os


class SessionCollector:
    """Collects and saves session data"""

    def __init__(self, user_height_cm, expected_leg_length_cm, user_id: str = None):
        """
        Initialize session collector.

        Args:
            user_height_cm: User's height in cm
            expected_leg_length_cm: Expected leg length for calibration
            user_id: Optional identifier for organizing sessions by user/test type
                    Examples: 'john', 'patient_001', 'test_run', 'calibration_check'
        """
        self.session_start = datetime.now()
        self.user_height = user_height_cm
        self.expected_leg = expected_leg_length_cm
        self.user_id = user_id or "anonymous"

        # Storage
        self.frames = []
        self.frame_count = 0

        # Session metadata
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        
    def add_frame(self, measurements, evaluations, smoothed, persistent_issues, trial_config=None):
        """
        Add frame data to the session.

        Args:
            measurements: dict with biomechanics measurements
            evaluations: dict with status/message for each metric
            smoothed: dict with temporally smoothed values
            persistent_issues: list of persistent issue names
            trial_config: optional dict from TrialManager.get_current_config()
                         If provided, merges trial info into measurements
        """
        self.frame_count += 1

        # Merge trial config into measurements if provided
        if trial_config:
            measurements = {
                **measurements,
                'trial_id': trial_config.get('trial_id'),
                'trial_index': trial_config.get('trial_index'),
                'crutch_model_id': trial_config.get('crutch_model_id'),
                'overall_length_cm': trial_config.get('overall_length_cm'),
                'grip_height_cm': trial_config.get('grip_height_cm'),
                'overall_setting_idx': trial_config.get('overall_setting_idx'),
                'grip_setting_idx': trial_config.get('grip_setting_idx'),
            }

        frame_data = {
            "frame_number": self.frame_count,
            "timestamp": (datetime.now() - self.session_start).total_seconds(),
            "measurements": measurements,
            "evaluations": {k: {"status": v["status"], "message": v["message"]}
                          for k, v in evaluations.items()},
            "smoothed_values": smoothed,
            "persistent_issues": persistent_issues
        }

        self.frames.append(frame_data)
    
    def save_session(self):
        """Save to JSON file in user-specific folder"""

        # Create user-specific sessions directory
        user_dir = os.path.join("sessions", self.user_id)
        os.makedirs(user_dir, exist_ok=True)

        # Session summary
        session_data = {
            "metadata": {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "start_time": self.session_start.isoformat(),
                "duration_seconds": (datetime.now() - self.session_start).total_seconds(),
                "total_frames": self.frame_count,
                "user_height_cm": self.user_height,
                "expected_leg_length_cm": self.expected_leg
            },
            "frames": self.frames
        }

        # Save to user folder
        filename = os.path.join(user_dir, f"session_{self.session_id}.json")
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)

        print(f"\n✓ Session saved: {filename}")
        print(f"  User: {self.user_id}")
        print(f"  Total frames: {self.frame_count}")
        print(f"  Duration: {session_data['metadata']['duration_seconds']:.1f}s")

        return filename