"""
Trial Manager - tracks crutch adjustments, user feedback, and recommendations
"""
import cv2
from typing import Optional, List, Dict


# Issue codes and their descriptions
ISSUE_CODES = {
    '1': {'code': 'too_tall', 'label': 'Crutches feel too TALL'},
    '2': {'code': 'too_short', 'label': 'Crutches feel too SHORT'},
    '3': {'code': 'handle_high', 'label': 'Handle/grip too HIGH'},
    '4': {'code': 'handle_low', 'label': 'Handle/grip too LOW'},
    '5': {'code': 'underarm_pressure', 'label': 'Underarm pressure/discomfort'},
    '6': {'code': 'wrist_strain', 'label': 'Wrist/hand strain'},
    '7': {'code': 'unstable', 'label': 'Felt unstable'},
    '8': {'code': 'fatigue', 'label': 'Got tired quickly'},
}


class TrialManager:
    """
    Manages crutch adjustment trials with user feedback and recommendations.

    Tracks:
    - Current crutch settings (overall length, grip height)
    - Trial history with objective measurements and subjective feedback
    - Best trial selection
    - Generates recommendations based on feedback patterns
    """

    def __init__(self, device_profile, initial_overall_cm: float, initial_grip_cm: float):
        """
        Initialize trial manager.

        Args:
            device_profile: CrutchDeviceProfile with device specifications
            initial_overall_cm: Starting overall crutch length in cm
            initial_grip_cm: Starting grip/handle height in cm
        """
        self.device = device_profile

        # Convert initial measurements to setting indices
        overall_setting = self.device.cm_to_setting(initial_overall_cm, 'overall')
        grip_setting = self.device.cm_to_setting(initial_grip_cm, 'grip')

        self.current_overall_idx = overall_setting['setting_idx']
        self.current_grip_idx = grip_setting['setting_idx']

        # Trial tracking
        self.current_trial_index = 0
        self.best_trial_index: Optional[int] = None
        self.trial_history: List[Dict] = []

        # Biomechanics data for current trial (updated by main loop)
        self.current_trial_measurements = {
            'elbow_angles': [],
            'trunk_lean': [],
            'issue_counts': {}
        }

        # Start first trial
        self.start_new_trial()

    @property
    def overall_cm(self) -> float:
        """Current overall length in cm"""
        return self.device.setting_to_cm(self.current_overall_idx, 'overall')

    @property
    def grip_cm(self) -> float:
        """Current grip height in cm"""
        return self.device.setting_to_cm(self.current_grip_idx, 'grip')

    def start_new_trial(self, collect_feedback: bool = False) -> Dict:
        """
        Start a new trial.

        Args:
            collect_feedback: If True, prompt for feedback on previous trial first

        Returns:
            dict with new trial configuration
        """
        # Collect feedback for previous trial if requested and not first trial
        if collect_feedback and self.current_trial_index > 0:
            self._collect_trial_feedback()

        self.current_trial_index += 1
        trial_id = f"T{self.current_trial_index:03d}"

        trial = {
            'trial_id': trial_id,
            'trial_index': self.current_trial_index,
            'overall_setting_idx': self.current_overall_idx,
            'grip_setting_idx': self.current_grip_idx,
            'overall_cm': self.overall_cm,
            'grip_cm': self.grip_cm,
            'overall_setting_number': self.current_overall_idx + 1,
            'grip_setting_number': self.current_grip_idx + 1,
            'crutch_model_id': self.device.model_id,
            'is_best': False,
            # Feedback fields (populated later)
            'user_rating': None,
            'user_issues': [],
            'biomechanics_summary': None,
            'recommendation_given': None
        }

        self.trial_history.append(trial)

        # Reset measurements for new trial
        self.current_trial_measurements = {
            'elbow_angles': [],
            'trunk_lean': [],
            'issue_counts': {}
        }

        print(f"\n{'='*50}")
        print(f"TRIAL {trial_id} STARTED")
        print(f"{'='*50}")
        print(f"  Overall: Hole #{self.current_overall_idx + 1} ({self.overall_cm:.1f} cm / {self.overall_cm/2.54:.1f} in)")
        print(f"  Grip:    Hole #{self.current_grip_idx + 1} ({self.grip_cm:.1f} cm / {self.grip_cm/2.54:.1f} in)")
        print(f"\nWalk for 30-60 seconds, then press 'N' for next trial")

        return trial

    def adjust_overall(self, direction: int) -> None:
        """
        Adjust overall crutch length by one hole position.

        Args:
            direction: +1 to raise (longer), -1 to lower (shorter)
        """
        new_idx = self.current_overall_idx + direction

        # Bounds checking
        max_idx = self.device.overall_num_settings - 1

        if new_idx < 0:
            print(f"\n⚠️  Already at MINIMUM overall length (Hole #1)")
            return
        elif new_idx > max_idx:
            print(f"\n⚠️  Already at MAXIMUM overall length (Hole #{max_idx + 1})")
            return

        self.current_overall_idx = new_idx
        new_cm = self.overall_cm

        direction_word = "RAISED" if direction > 0 else "LOWERED"
        print(f"\n📏 Overall {direction_word}: Hole #{new_idx + 1} ({new_cm:.1f} cm / {new_cm/2.54:.1f} in)")

    def adjust_grip(self, direction: int) -> None:
        """
        Adjust grip/handle height by one hole position.

        Args:
            direction: +1 to raise, -1 to lower
        """
        new_idx = self.current_grip_idx + direction

        # Bounds checking
        max_idx = self.device.grip_num_settings - 1

        if new_idx < 0:
            print(f"\n⚠️  Already at MINIMUM grip height (Hole #1)")
            return
        elif new_idx > max_idx:
            print(f"\n⚠️  Already at MAXIMUM grip height (Hole #{max_idx + 1})")
            return

        self.current_grip_idx = new_idx
        new_cm = self.grip_cm

        direction_word = "RAISED" if direction > 0 else "LOWERED"
        print(f"\n✋ Grip {direction_word}: Hole #{new_idx + 1} ({new_cm:.1f} cm / {new_cm/2.54:.1f} in)")

    def mark_as_best(self) -> None:
        """Mark current trial as the best one."""
        self.best_trial_index = self.current_trial_index

        for trial in self.trial_history:
            trial['is_best'] = (trial['trial_index'] == self.current_trial_index)

        print(f"\n⭐ Trial {self.current_trial_index} marked as BEST!")

    def get_current_config(self) -> Dict:
        """
        Get current trial configuration.

        Returns:
            dict with trial_id, settings, and status
        """
        current_trial = self.trial_history[-1] if self.trial_history else {}

        return {
            'trial_id': current_trial.get('trial_id', 'T001'),
            'trial_index': self.current_trial_index,
            'crutch_model_id': self.device.model_id,
            'overall_length_cm': self.overall_cm,
            'grip_height_cm': self.grip_cm,
            'overall_setting_idx': self.current_overall_idx,
            'grip_setting_idx': self.current_grip_idx,
            'is_best': current_trial.get('is_best', False)
        }

    def update_biomechanics(self, elbow_angle: float, trunk_lean: float, issues: List[str]) -> None:
        """
        Update biomechanics data for current trial (called each frame).

        Args:
            elbow_angle: Average elbow angle this frame
            trunk_lean: Trunk lean angle this frame
            issues: List of issue codes detected this frame
        """
        self.current_trial_measurements['elbow_angles'].append(elbow_angle)
        self.current_trial_measurements['trunk_lean'].append(trunk_lean)

        for issue in issues:
            self.current_trial_measurements['issue_counts'][issue] = \
                self.current_trial_measurements['issue_counts'].get(issue, 0) + 1

    def _collect_trial_feedback(self) -> None:
        """Collect user feedback for the current trial via terminal prompts."""
        if not self.trial_history:
            return

        current_trial = self.trial_history[-1]
        trial_id = current_trial['trial_id']

        print(f"\n{'='*50}")
        print(f"FEEDBACK FOR {trial_id}")
        print(f"{'='*50}")

        # Collect rating
        while True:
            rating_input = input("\nRate this trial (1-5, or Enter to skip): ").strip()
            if rating_input == '':
                rating = None
                break
            try:
                rating = int(rating_input)
                if 1 <= rating <= 5:
                    break
                else:
                    print("Please enter a number between 1 and 5")
            except ValueError:
                print("Please enter a number between 1 and 5")

        current_trial['user_rating'] = rating

        # Collect issues
        print("\nAny issues? Select all that apply:")
        for key, info in ISSUE_CODES.items():
            print(f"  [{key}] {info['label']}")
        print("  [Enter] No issues / skip")

        issues_input = input("\nIssues (e.g., '1,5' or Enter): ").strip()

        issues = []
        if issues_input:
            for char in issues_input.replace(',', ' ').replace(';', ' ').split():
                char = char.strip()
                if char in ISSUE_CODES:
                    issues.append(ISSUE_CODES[char]['code'])

        current_trial['user_issues'] = issues

        # Summarize biomechanics
        if self.current_trial_measurements['elbow_angles']:
            import numpy as np
            avg_elbow = np.mean(self.current_trial_measurements['elbow_angles'])
            avg_trunk = np.mean(self.current_trial_measurements['trunk_lean'])
            current_trial['biomechanics_summary'] = {
                'avg_elbow_angle': round(avg_elbow, 1),
                'avg_trunk_lean': round(avg_trunk, 1),
                'persistent_issues': self.current_trial_measurements['issue_counts']
            }

        # Generate recommendation
        recommendation = self._generate_recommendation(current_trial)
        current_trial['recommendation_given'] = recommendation

        if recommendation:
            print(f"\n💡 SUGGESTION: {recommendation}")

    def _generate_recommendation(self, trial: Dict) -> Optional[str]:
        """
        Generate a recommendation based on trial feedback and biomechanics.

        Args:
            trial: Trial dict with feedback data

        Returns:
            Recommendation string or None
        """
        issues = trial.get('user_issues', [])
        rating = trial.get('user_rating')
        bio = trial.get('biomechanics_summary', {})

        recommendations = []

        # Check for specific issue patterns
        if 'too_tall' in issues or 'underarm_pressure' in issues:
            recommendations.append("Try LOWERING overall length by 1-2 holes")

        if 'too_short' in issues:
            recommendations.append("Try RAISING overall length by 1-2 holes")

        if 'handle_high' in issues:
            recommendations.append("Try LOWERING grip by 1 hole")

        if 'handle_low' in issues or 'wrist_strain' in issues:
            recommendations.append("Try RAISING grip by 1 hole")

        if 'unstable' in issues:
            recommendations.append("Check overall height - instability often means crutches are too tall")

        # Check biomechanics
        if bio:
            avg_elbow = bio.get('avg_elbow_angle', 25)
            if avg_elbow > 35:
                recommendations.append("Elbow angle high (>35°) - consider lowering grip")
            elif avg_elbow < 15:
                recommendations.append("Elbow angle low (<15°) - consider raising grip")

        # Check improvement from previous trial
        if len(self.trial_history) >= 2:
            prev_trial = self.trial_history[-2]
            prev_rating = prev_trial.get('user_rating')

            if rating and prev_rating:
                if rating > prev_rating:
                    # Settings improved comfort
                    overall_delta = trial['overall_setting_idx'] - prev_trial['overall_setting_idx']
                    grip_delta = trial['grip_setting_idx'] - prev_trial['grip_setting_idx']

                    if overall_delta != 0 or grip_delta != 0:
                        recommendations.insert(0, f"Good progress! Rating improved from {prev_rating} to {rating}")
                elif rating < prev_rating:
                    recommendations.insert(0, f"Rating dropped from {prev_rating} to {rating} - consider reverting last change")

        # Good rating with no issues
        if rating and rating >= 4 and not issues:
            return "Settings look good! Try one more trial to confirm, or mark as BEST (press B)"

        if recommendations:
            return " | ".join(recommendations[:2])  # Limit to 2 suggestions

        return None

    def end_trial_with_feedback(self) -> None:
        """End current trial and collect feedback before starting new one."""
        self._collect_trial_feedback()

    def draw_config_overlay(self, image) -> None:
        """
        Draw trial configuration overlay on the video frame.

        Args:
            image: OpenCV image (numpy array)
        """
        height, width = image.shape[:2]

        current_trial = self.trial_history[-1] if self.trial_history else {}

        # === TOP RIGHT: Trial info box ===
        info_lines = [
            f"Trial: {current_trial.get('trial_id', 'T001')}",
            f"Overall: Hole #{self.current_overall_idx + 1} ({self.overall_cm:.0f}cm)",
            f"Grip: Hole #{self.current_grip_idx + 1} ({self.grip_cm:.0f}cm)",
        ]

        # Add rating if available
        if current_trial.get('user_rating'):
            stars = '*' * current_trial['user_rating']
            info_lines.append(f"Rating: {stars}")

        box_x = width - 220
        box_y = 10
        box_w = 210
        line_height = 22
        box_h = len(info_lines) * line_height + 15

        # Background
        overlay = image.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h),
                     (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)

        # Border
        border_color = (0, 255, 255) if current_trial.get('is_best') else (100, 100, 100)
        cv2.rectangle(image, (box_x, box_y), (box_x + box_w, box_y + box_h),
                     border_color, 2)

        # Text
        for i, line in enumerate(info_lines):
            y = box_y + 18 + (i * line_height)
            cv2.putText(image, line, (box_x + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        # Best badge
        if current_trial.get('is_best'):
            badge_y = box_y + box_h + 25
            cv2.putText(image, "BEST", (box_x + 80, badge_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # === BOTTOM: Controls help ===
        controls_y = height - 15
        controls_text = "[/]: grip  -/=: overall  N: new trial  B: mark best  E: end+feedback"

        # Background bar
        cv2.rectangle(image, (0, height - 30), (width, height), (30, 30, 30), -1)
        cv2.putText(image, controls_text, (10, controls_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    def print_session_summary(self) -> Dict:
        """
        Print end-of-session summary with all trials and recommendation.

        Returns:
            dict with summary data
        """
        print(f"\n{'='*60}")
        print("SESSION SUMMARY")
        print(f"{'='*60}")

        if not self.trial_history:
            print("No trials recorded.")
            return {}

        best_trial = None
        highest_rated_trial = None
        highest_rating = 0

        for trial in self.trial_history:
            trial_id = trial['trial_id']
            rating = trial.get('user_rating')
            issues = trial.get('user_issues', [])
            is_best = trial.get('is_best', False)

            # Star display
            if rating:
                stars = '★' * rating + '☆' * (5 - rating)
                rating_str = f"{stars} ({rating}/5)"
            else:
                stars = ''
                rating_str = "Not rated"

            # Best marker
            best_marker = " ⭐ BEST" if is_best else ""
            if is_best:
                best_trial = trial

            # Track highest rated
            if rating and rating > highest_rating:
                highest_rating = rating
                highest_rated_trial = trial

            print(f"\n{trial_id}: {rating_str}{best_marker}")
            print(f"  Settings: Overall hole #{trial['overall_setting_idx'] + 1}, Grip hole #{trial['grip_setting_idx'] + 1}")
            print(f"            ({trial['overall_cm']:.0f}cm overall, {trial['grip_cm']:.0f}cm grip)")

            if issues:
                issue_labels = [ISSUE_CODES.get(i, {}).get('label', i) for i in issues
                               if isinstance(i, str) and i in [v['code'] for v in ISSUE_CODES.values()]]
                # Handle issue codes directly
                issue_labels = []
                for issue_code in issues:
                    for info in ISSUE_CODES.values():
                        if info['code'] == issue_code:
                            issue_labels.append(info['label'])
                            break
                if issue_labels:
                    print(f"  Issues: {', '.join(issue_labels)}")
            else:
                print(f"  Issues: None reported")

        # Final recommendation
        print(f"\n{'-'*60}")
        print("RECOMMENDATION")
        print(f"{'-'*60}")

        recommended_trial = best_trial or highest_rated_trial

        if recommended_trial:
            trial_id = recommended_trial['trial_id']
            reason = "marked as BEST by user" if recommended_trial.get('is_best') else "highest comfort rating"

            print(f"\nUse {trial_id} settings ({reason}):")
            print(f"  Overall: Hole #{recommended_trial['overall_setting_idx'] + 1} ({recommended_trial['overall_cm']:.0f} cm)")
            print(f"  Grip:    Hole #{recommended_trial['grip_setting_idx'] + 1} ({recommended_trial['grip_cm']:.0f} cm)")

            # Compare to first trial
            first_trial = self.trial_history[0]
            overall_change = recommended_trial['overall_setting_idx'] - first_trial['overall_setting_idx']
            grip_change = recommended_trial['grip_setting_idx'] - first_trial['grip_setting_idx']

            if overall_change != 0 or grip_change != 0:
                print(f"\nChanges from initial settings:")
                if overall_change != 0:
                    direction = "raised" if overall_change > 0 else "lowered"
                    print(f"  Overall: {direction} by {abs(overall_change)} hole(s)")
                if grip_change != 0:
                    direction = "raised" if grip_change > 0 else "lowered"
                    print(f"  Grip: {direction} by {abs(grip_change)} hole(s)")
        else:
            print("\nNo clear recommendation - try more trials with different settings")

        return {
            'total_trials': len(self.trial_history),
            'best_trial': best_trial,
            'highest_rated_trial': highest_rated_trial,
            'recommended_trial': recommended_trial,
            'trial_history': self.trial_history
        }

    def get_adjustment_suggestion(self) -> Optional[str]:
        """
        Get a real-time suggestion based on current trial's biomechanics.

        Returns:
            Suggestion string or None
        """
        if not self.current_trial_measurements['elbow_angles']:
            return None

        import numpy as np
        recent_elbow = self.current_trial_measurements['elbow_angles'][-30:]  # Last ~1 second

        if len(recent_elbow) < 10:
            return None

        avg_elbow = np.mean(recent_elbow)

        if avg_elbow > 40:
            return "Elbows very bent - grip may be too high"
        elif avg_elbow < 10:
            return "Arms too straight - grip may be too low"

        return None
