"""
Configuration and device profiles
"""
from collections import deque

# MediaPipe pose landmark indices (for quick access)
class PoseLandmarks:
    NOSE = 0
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


class CrutchDeviceProfile:
    """Stores specifications for a specific crutch model"""

    def __init__(self, model_id, model_name, crutch_type='axillary',
                 overall_min_cm=110, overall_max_cm=145, overall_step_cm=2.54,
                 grip_min_cm=70, grip_max_cm=100, grip_step_cm=2.54,
                 underarm_clearance_cm=5.0):
        self.model_id = model_id
        self.model_name = model_name
        self.crutch_type = crutch_type

        # Overall length settings (main tube)
        self.overall_step_cm = overall_step_cm
        self.overall_min_cm = overall_min_cm
        self.overall_max_cm = overall_max_cm
        self.overall_num_settings = int((self.overall_max_cm - self.overall_min_cm) / self.overall_step_cm) + 1

        # Grip/handle height settings
        self.grip_step_cm = grip_step_cm
        self.grip_min_cm = grip_min_cm
        self.grip_max_cm = grip_max_cm
        self.grip_num_settings = int((self.grip_max_cm - self.grip_min_cm) / self.grip_step_cm) + 1

        # Clinical target: 2 inches (5cm) clearance from underarm
        self.underarm_clearance_cm = underarm_clearance_cm

    @classmethod
    def from_scan(cls, scan_data, current_length_cm=None):
        """
        Create a profile from scanned crutch data.

        Args:
            scan_data: dict from hole_scanner with keys:
                - hole_count: number of adjustment holes
                - hole_spacing_cm: distance between holes
                - adjustable_range_cm: total range
                - handle_holes: (optional) handle adjustment holes
                - handle_spacing_cm: (optional) handle hole spacing
            current_length_cm: current overall length (used to estimate min/max)

        Returns:
            CrutchDeviceProfile configured from scanned values
        """
        hole_count = scan_data.get('hole_count', 14)
        hole_spacing = scan_data.get('hole_spacing_cm', 2.54)
        adjustable_range = scan_data.get('adjustable_range_cm')

        # Calculate adjustable range if not provided
        if adjustable_range is None:
            adjustable_range = (hole_count - 1) * hole_spacing

        # Estimate min/max based on current length and range
        # Assume current setting is roughly in the middle
        if current_length_cm:
            overall_min = current_length_cm - (adjustable_range / 2)
            overall_max = current_length_cm + (adjustable_range / 2)
        else:
            # Default to typical adult range
            overall_min = 110
            overall_max = overall_min + adjustable_range

        # Handle/grip settings
        handle_holes = scan_data.get('handle_holes')
        handle_spacing = scan_data.get('handle_spacing_cm', 2.54)

        if handle_holes and handle_holes > 1:
            grip_range = (handle_holes - 1) * handle_spacing
            grip_min = 70  # Typical minimum
            grip_max = grip_min + grip_range
        else:
            # Default grip range
            grip_min = 70
            grip_max = 100
            handle_spacing = 2.54

        return cls(
            model_id="scanned_crutch",
            model_name="Scanned Crutch",
            crutch_type='axillary',
            overall_min_cm=overall_min,
            overall_max_cm=overall_max,
            overall_step_cm=hole_spacing,
            grip_min_cm=grip_min,
            grip_max_cm=grip_max,
            grip_step_cm=handle_spacing
        )
    
    def cm_to_setting(self, measurement_cm, adjustment_type='overall'):
        """Convert cm measurement to nearest device setting"""
        if adjustment_type == 'overall':
            min_cm, step_cm = self.overall_min_cm, self.overall_step_cm
            max_cm = self.overall_max_cm
        elif adjustment_type == 'grip':
            min_cm, step_cm = self.grip_min_cm, self.grip_step_cm
            max_cm = self.grip_max_cm
        else:
            raise ValueError(f"Unknown type: {adjustment_type}")
        
        measurement_cm = max(min_cm, min(max_cm, measurement_cm))
        setting_idx = round((measurement_cm - min_cm) / step_cm)
        actual_cm = min_cm + (setting_idx * step_cm)
        
        return {
            'setting_idx': setting_idx,
            'setting_number': setting_idx + 1,
            'actual_cm': actual_cm,
            'actual_inches': actual_cm / 2.54
        }
    
    def setting_to_cm(self, setting_idx, adjustment_type='overall'):
        """Convert setting index to cm"""
        if adjustment_type == 'overall':
            return self.overall_min_cm + (setting_idx * self.overall_step_cm)
        elif adjustment_type == 'grip':
            return self.grip_min_cm + (setting_idx * self.grip_step_cm)
    
    def get_info(self):
        """Return profile summary"""
        return {
            'model_id': self.model_id,
            'model_name': self.model_name,
            'type': self.crutch_type,
            'overall_settings': f"{self.overall_num_settings} positions ({self.overall_min_cm:.0f}-{self.overall_max_cm:.0f}cm)",
            'grip_settings': f"{self.grip_num_settings} holes ({self.grip_min_cm:.0f}-{self.grip_max_cm:.0f}cm)"
        }


# Default device profile
DEFAULT_CRUTCH = CrutchDeviceProfile(
    model_id="generic_axillary_adult",
    model_name="Standard Adult Axillary Crutch",
    crutch_type='axillary'
)