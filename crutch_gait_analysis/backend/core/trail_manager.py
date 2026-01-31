"""
Trial management for multi-trial fitting sessions
"""
import cv2


class TrialManager:
    """Manages trial tracking during fitting sessions"""
    
    def __init__(self, device_profile, initial_overall_cm, initial_grip_cm):
        self.device = device_profile
        self.current_trial_index = 1
        self.best_trial_index = None
        
        # Convert initial settings to indices
        overall_info = device_profile.cm_to_setting(initial_overall_cm, 'overall')
        grip_info = device_profile.cm_to_setting(initial_grip_cm, 'grip')
        
        self.current_overall_idx = overall_info['setting_idx']
        self.current_grip_idx = grip_info['setting_idx']
        
        print(f"\n✓ Trial Manager initialized")
        print(f"  Starting Trial: 1")
        print(f"  Grip: Hole #{grip_info['setting_number']} ({grip_info['actual_cm']:.0f}cm)")
        print(f"  Overall: Setting #{overall_info['setting_number']} ({overall_info['actual_cm']:.0f}cm)")
    
    def start_new_trial(self):
        """Mark start of new trial"""
        self.current_trial_index += 1
        print(f"\n🆕 Started Trial {self.current_trial_index}")
    
    def mark_as_best(self):
        """Mark current trial as best"""
        self.best_trial_index = self.current_trial_index
        print(f"⭐ Trial {self.current_trial_index} marked as BEST")
    
    def adjust_grip(self, direction):
        """Adjust grip setting up/down"""
        new_idx = self.current_grip_idx + direction
        
        max_idx = self.device.grip_num_settings - 1
        if 0 <= new_idx <= max_idx:
            self.current_grip_idx = new_idx
            new_cm = self.device.setting_to_cm(new_idx, 'grip')
            print(f"🔧 Grip: Hole #{new_idx + 1} ({new_cm:.1f}cm / {new_cm/2.54:.1f}\")")
        else:
            print(f"⚠️  Grip already at {'MIN' if direction < 0 else 'MAX'}")
    
    def adjust_overall(self, direction):
        """Adjust overall height up/down"""
        new_idx = self.current_overall_idx + direction
        
        max_idx = self.device.overall_num_settings - 1
        if 0 <= new_idx <= max_idx:
            self.current_overall_idx = new_idx
            new_cm = self.device.setting_to_cm(new_idx, 'overall')
            print(f"🔧 Overall: Setting #{new_idx + 1} ({new_cm:.1f}cm / {new_cm/2.54:.1f}\")")
        else:
            print(f"⚠️  Overall already at {'MIN' if direction < 0 else 'MAX'}")
    
    def get_current_config(self):
        """Get current trial configuration"""
        return {
            'trial_id': f"T{self.current_trial_index:03d}",
            'trial_index': self.current_trial_index,
            'crutch_model_id': self.device.model_id,
            'overall_length_cm': self.device.setting_to_cm(self.current_overall_idx, 'overall'),
            'grip_height_cm': self.device.setting_to_cm(self.current_grip_idx, 'grip'),
            'overall_setting_idx': self.current_overall_idx,
            'grip_setting_idx': self.current_grip_idx,
            'is_best': (self.current_trial_index == self.best_trial_index)
        }
    
    def draw_config_overlay(self, image):
        """Draw current configuration on frame"""
        config = self.get_current_config()
        h, w = image.shape[:2]
        
        # Semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (10, h - 120), (420, h - 10), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Trial info
        trial_text = f"Trial {config['trial_index']}"
        if config['is_best']:
            trial_text += " ⭐ BEST"
            color = (0, 255, 255)
        else:
            color = (255, 255, 255)
        
        cv2.putText(image, trial_text, (20, h - 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Settings
        grip_text = f"Grip: #{config['grip_setting_idx'] + 1} ({config['grip_height_cm']:.0f}cm)"
        cv2.putText(image, grip_text, (20, h - 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        overall_text = f"Overall: #{config['overall_setting_idx'] + 1} ({config['overall_length_cm']:.0f}cm)"
        cv2.putText(image, overall_text, (20, h - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Controls
        controls = "[/]: grip  -/=: overall  N: new  B: best"
        cv2.putText(image, controls, (20, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)