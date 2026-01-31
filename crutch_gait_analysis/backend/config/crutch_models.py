#!/usr/bin/env python3
"""
Crutch Device Profiles

Defines physical characteristics of crutch models including:
- Discrete adjustment settings (hole positions)
- Height ranges
- Conversion between cm and setting indices

Standard forearm crutches have:
- Overall length adjustment (bottom section): typically 5-8 positions
- Grip height adjustment (arm cuff section): typically 8-12 hole positions
- Each position is typically 1 inch (2.54 cm) apart
"""

import numpy as np
from typing import List, Dict, Optional


class CrutchDeviceProfile:
    """
    Represents a specific crutch model with its adjustment ranges.

    Crutches have discrete adjustment positions (holes), not continuous
    adjustment. This class handles conversion between desired cm values
    and actual available settings.

    Attributes:
        name: Model name
        grip_settings_cm: List of available grip heights in cm
        overall_settings_cm: List of available overall lengths in cm

    Example:
        >>> crutch = CrutchDeviceProfile("Standard", grip_range=(76, 96), ...)
        >>> info = crutch.cm_to_setting(85.0, 'grip')
        >>> print(f"Setting #{info['setting_number']}: {info['actual_cm']}cm")
    """

    def __init__(self,
                 name: str,
                 grip_range: tuple,
                 grip_positions: int,
                 overall_range: tuple,
                 overall_positions: int,
                 model_id: str = None):
        """
        Initialize crutch profile.

        Args:
            name: Model identifier
            grip_range: (min_cm, max_cm) for grip height
            grip_positions: Number of discrete grip positions
            overall_range: (min_cm, max_cm) for overall length
            overall_positions: Number of discrete overall positions
            model_id: Unique model ID (defaults to lowercase name)
        """
        self.name = name
        self.model_name = name  # Alias for compatibility
        self.model_id = model_id or name.lower().replace(' ', '_')

        # Generate discrete settings
        self.grip_settings_cm = list(np.linspace(
            grip_range[0], grip_range[1], grip_positions
        ))
        self.overall_settings_cm = list(np.linspace(
            overall_range[0], overall_range[1], overall_positions
        ))

        # Store ranges
        self.grip_range = grip_range
        self.overall_range = overall_range

        # Store counts for easy access
        self.grip_num_settings = grip_positions
        self.overall_num_settings = overall_positions

        # Calculate step sizes
        self.grip_step_cm = (grip_range[1] - grip_range[0]) / (grip_positions - 1) if grip_positions > 1 else 0
        self.overall_step_cm = (overall_range[1] - overall_range[0]) / (overall_positions - 1) if overall_positions > 1 else 0

    def cm_to_setting(self, target_cm: float, setting_type: str) -> Dict:
        """
        Convert target cm value to nearest available setting.

        Args:
            target_cm: Desired height/length in cm
            setting_type: 'grip' or 'overall'

        Returns:
            dict with:
                - setting_idx: 0-based index
                - setting_number: 1-based number (for display)
                - actual_cm: Actual cm value at that setting
                - target_cm: Original requested value
                - difference_cm: How far actual is from target
        """
        if setting_type == 'grip':
            settings = self.grip_settings_cm
        elif setting_type == 'overall':
            settings = self.overall_settings_cm
        else:
            raise ValueError(f"Unknown setting_type: {setting_type}")

        # Find nearest setting
        differences = [abs(s - target_cm) for s in settings]
        idx = differences.index(min(differences))

        return {
            'setting_idx': idx,
            'setting_number': idx + 1,  # 1-based for display
            'actual_cm': round(settings[idx], 2),
            'target_cm': target_cm,
            'difference_cm': round(settings[idx] - target_cm, 2)
        }

    def setting_to_cm(self, setting_idx: int, setting_type: str) -> float:
        """
        Get cm value for a specific setting index.

        Args:
            setting_idx: 0-based setting index
            setting_type: 'grip' or 'overall'

        Returns:
            Height/length in cm at that setting
        """
        if setting_type == 'grip':
            settings = self.grip_settings_cm
        elif setting_type == 'overall':
            settings = self.overall_settings_cm
        else:
            raise ValueError(f"Unknown setting_type: {setting_type}")

        if setting_idx < 0 or setting_idx >= len(settings):
            raise IndexError(f"Setting index {setting_idx} out of range [0, {len(settings)-1}]")

        return settings[setting_idx]

    def get_all_settings(self, setting_type: str) -> List[Dict]:
        """
        Get all available settings for a setting type.

        Args:
            setting_type: 'grip' or 'overall'

        Returns:
            List of dicts with setting_idx, setting_number, and cm value
        """
        if setting_type == 'grip':
            settings = self.grip_settings_cm
        elif setting_type == 'overall':
            settings = self.overall_settings_cm
        else:
            raise ValueError(f"Unknown setting_type: {setting_type}")

        return [
            {
                'setting_idx': i,
                'setting_number': i + 1,
                'cm': round(s, 2),
                'inches': round(s / 2.54, 2)
            }
            for i, s in enumerate(settings)
        ]

    def print_settings(self):
        """Print all available settings for this crutch model."""
        print(f"\n📐 {self.name} CRUTCH SETTINGS")
        print("=" * 50)

        print(f"\nGrip Height Positions ({len(self.grip_settings_cm)} holes):")
        for i, cm in enumerate(self.grip_settings_cm):
            print(f"   Hole #{i+1}: {cm:.1f}cm ({cm/2.54:.1f}\")")

        print(f"\nOverall Length Positions ({len(self.overall_settings_cm)} settings):")
        for i, cm in enumerate(self.overall_settings_cm):
            print(f"   Setting #{i+1}: {cm:.1f}cm ({cm/2.54:.1f}\")")

    def __repr__(self):
        return (f"CrutchDeviceProfile('{self.name}', "
                f"grip={len(self.grip_settings_cm)} positions, "
                f"overall={len(self.overall_settings_cm)} positions)")

    @classmethod
    def from_scan(cls, scan_data: dict, current_length_cm: float = None):
        """
        Create a CrutchDeviceProfile from scanned crutch data.

        Args:
            scan_data: Dictionary from crutch scanner containing:
                - hole_count: Number of adjustment holes detected
                - hole_spacing_cm: Distance between holes in cm
                - length_cm: Total crutch length
                - handle_height_cm: Current handle height
            current_length_cm: Current overall length (optional)

        Returns:
            CrutchDeviceProfile configured from scan data
        """
        hole_count = scan_data.get('hole_count', 8)
        hole_spacing = scan_data.get('hole_spacing_cm', 2.54)
        length_cm = scan_data.get('length_cm', current_length_cm or 125.0)
        handle_height = scan_data.get('handle_height_cm', 85.0)

        # Estimate grip range based on detected holes
        grip_range_cm = (hole_count - 1) * hole_spacing
        grip_min = handle_height - (grip_range_cm / 2)
        grip_max = handle_height + (grip_range_cm / 2)

        # Estimate overall range (typically 6 positions, ~3.8cm apart)
        overall_positions = 6
        overall_spacing = 3.8  # ~1.5 inches
        overall_range_cm = (overall_positions - 1) * overall_spacing
        overall_min = length_cm - (overall_range_cm / 2)
        overall_max = length_cm + (overall_range_cm / 2)

        return cls(
            name="Scanned Crutch",
            grip_range=(grip_min, grip_max),
            grip_positions=hole_count,
            overall_range=(overall_min, overall_max),
            overall_positions=overall_positions,
            model_id="scanned"
        )


# =============================================================================
# STANDARD CRUTCH PROFILES
# =============================================================================

# Standard adult forearm crutch
# Based on typical aluminum forearm crutch specifications
# Grip: 8 holes, ~1 inch apart
# Overall: 6 positions, ~1.5 inches apart

DEFAULT_CRUTCH = CrutchDeviceProfile(
    name="Standard Adult Forearm Crutch",
    grip_range=(76.2, 94.0),      # 30" to 37" (typical range)
    grip_positions=8,              # 8 hole positions
    overall_range=(114.3, 137.2),  # 45" to 54" (typical range)
    overall_positions=6            # 6 push-button positions
)

# Tall adult forearm crutch
TALL_CRUTCH = CrutchDeviceProfile(
    name="Tall Adult Forearm Crutch",
    grip_range=(81.3, 101.6),     # 32" to 40"
    grip_positions=9,
    overall_range=(121.9, 149.9),  # 48" to 59"
    overall_positions=7
)

# Pediatric forearm crutch
PEDIATRIC_CRUTCH = CrutchDeviceProfile(
    name="Pediatric Forearm Crutch",
    grip_range=(55.9, 73.7),      # 22" to 29"
    grip_positions=8,
    overall_range=(73.7, 96.5),   # 29" to 38"
    overall_positions=6
)

# All available profiles
CRUTCH_PROFILES = {
    'standard': DEFAULT_CRUTCH,
    'tall': TALL_CRUTCH,
    'pediatric': PEDIATRIC_CRUTCH,
}


def get_profile(name: str) -> CrutchDeviceProfile:
    """
    Get a crutch profile by name.

    Args:
        name: Profile name ('standard', 'tall', 'pediatric')

    Returns:
        CrutchDeviceProfile instance

    Raises:
        KeyError: If profile name not found
    """
    if name not in CRUTCH_PROFILES:
        available = ', '.join(CRUTCH_PROFILES.keys())
        raise KeyError(f"Unknown profile '{name}'. Available: {available}")
    return CRUTCH_PROFILES[name]


# Quick test when run directly
if __name__ == '__main__':
    print("Testing CrutchDeviceProfile...")

    crutch = DEFAULT_CRUTCH
    crutch.print_settings()

    print("\n\nConversion Tests:")
    print("-" * 40)

    test_values = [80, 85, 90]
    for cm in test_values:
        info = crutch.cm_to_setting(cm, 'grip')
        print(f"Target {cm}cm → Hole #{info['setting_number']} "
              f"({info['actual_cm']}cm, diff: {info['difference_cm']:+.1f}cm)")

    test_values = [120, 125, 130]
    for cm in test_values:
        info = crutch.cm_to_setting(cm, 'overall')
        print(f"Target {cm}cm → Setting #{info['setting_number']} "
              f"({info['actual_cm']}cm, diff: {info['difference_cm']:+.1f}cm)")
