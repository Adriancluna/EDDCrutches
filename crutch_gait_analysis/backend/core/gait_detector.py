"""
Gait phase detection
"""
from collections import deque
import numpy as np


class GaitPhaseDetector:
    """
    Detect gait phases based on movement patterns.
    
    Phases:
    - STANDING: Both feet stationary, minimal trunk movement
    - WEIGHT_BEARING_LEFT: Left foot planted, right foot swinging
    - WEIGHT_BEARING_RIGHT: Right foot planted, left foot swinging  
    - DOUBLE_SUPPORT: Both feet planted (transition)
    - SWING_PHASE: Active walking motion
    """
    
    def __init__(self, history_size=15):
        """
        Args:
            history_size: Number of frames to track for velocity calculation
        """
        self.history_size = history_size
        
        # Position history for velocity calculation
        self.ankle_history_r = deque(maxlen=history_size)
        self.ankle_history_l = deque(maxlen=history_size)
        self.hip_history = deque(maxlen=history_size)
        
        # Phase history for smoothing
        self.phase_history = deque(maxlen=10)
        
        # Current phase
        self.current_phase = "STANDING"
        
    def update(self, landmarks_dict):
        """
        Update detector with new frame data.
        
        Args:
            landmarks_dict: Dict with keys like 'rx_ankle', 'lx_ankle', etc.
        """
        # Store positions
        self.ankle_history_r.append({
            'x': landmarks_dict['rx_ankle'],
            'y': landmarks_dict['ry_ankle']
        })
        self.ankle_history_l.append({
            'x': landmarks_dict['lx_ankle'],
            'y': landmarks_dict['ly_ankle']
        })
        
        # Mid-hip position (for trunk movement)
        mid_hip_x = (landmarks_dict['lx_hip'] + landmarks_dict['rx_hip']) / 2
        mid_hip_y = (landmarks_dict['ly_hip'] + landmarks_dict['ry_hip']) / 2
        self.hip_history.append({'x': mid_hip_x, 'y': mid_hip_y})
        
        # Detect phase if we have enough history
        if len(self.ankle_history_r) >= self.history_size // 2:
            self.current_phase = self._classify_phase(landmarks_dict)
            self.phase_history.append(self.current_phase)
    
    def _classify_phase(self, landmarks_dict):
        """Classify current gait phase based on movement patterns"""
        
        # Calculate velocities (movement over last few frames)
        velocity_r = self._calculate_velocity(self.ankle_history_r)
        velocity_l = self._calculate_velocity(self.ankle_history_l)
        hip_velocity = self._calculate_velocity(self.hip_history)
        
        # Calculate foot heights (relative to hip)
        mid_hip_y = (landmarks_dict['ly_hip'] + landmarks_dict['ry_hip']) / 2
        foot_height_r = mid_hip_y - landmarks_dict['ry_foot']  # Positive = foot below hip
        foot_height_l = mid_hip_y - landmarks_dict['ly_foot']
        
        # Detect if feet are lifted
        foot_lifted_r = foot_height_r < foot_height_l - 20  # Right foot higher
        foot_lifted_l = foot_height_l < foot_height_r - 20  # Left foot higher
        
        # Overall movement intensity
        total_movement = velocity_r + velocity_l + hip_velocity
        
        # ========================================
        # PHASE CLASSIFICATION LOGIC
        # ========================================
        
        # STANDING: Very little movement
        if total_movement < 3.0:
            return "STANDING"
        
        # WEIGHT_BEARING phases: One foot moving significantly more
        if velocity_r > velocity_l * 1.5 and velocity_r > 4.0:
            # Right foot moving more = swinging right = weight on left
            return "WEIGHT_BEARING_LEFT"
        
        if velocity_l > velocity_r * 1.5 and velocity_l > 4.0:
            # Left foot moving more = swinging left = weight on right
            return "WEIGHT_BEARING_RIGHT"
        
        # Check foot lift (more reliable than velocity sometimes)
        if foot_lifted_r and not foot_lifted_l:
            return "WEIGHT_BEARING_LEFT"  # Right foot lifted
        
        if foot_lifted_l and not foot_lifted_r:
            return "WEIGHT_BEARING_RIGHT"  # Left foot lifted
        
        # DOUBLE_SUPPORT: Both feet moving roughly equally at moderate speed
        if abs(velocity_r - velocity_l) < 2.0 and total_movement > 5.0:
            return "DOUBLE_SUPPORT"
        
        # SWING_PHASE: High overall movement, both feet active
        if total_movement > 8.0:
            return "SWING_PHASE"
        
        # Default to previous phase if unclear (smoothing)
        if self.phase_history:
            return self.phase_history[-1]
        
        return "STANDING"
    
    def _calculate_velocity(self, position_history):
        """
        Calculate velocity magnitude from position history.
        
        Returns: Average pixel movement per frame over recent history
        """
        if len(position_history) < 5:
            return 0.0
        
        # Compare recent position to 5 frames ago
        recent = list(position_history)[-1]
        past = list(position_history)[-5]
        
        dx = recent['x'] - past['x']
        dy = recent['y'] - past['y']
        
        # Distance per frame
        distance = np.sqrt(dx**2 + dy**2)
        velocity = distance / 5.0  # Average over 5 frames
        
        return velocity
    
    def get_phase(self):
        """Get current smoothed phase"""
        if not self.phase_history:
            return "STANDING"
        
        # Use majority voting over last few frames for stability
        recent_phases = list(self.phase_history)[-5:]
        
        # Count occurrences
        phase_counts = {}
        for phase in recent_phases:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        # Return most common
        return max(phase_counts.items(), key=lambda x: x[1])[0]
    
    def get_phase_confidence(self):
        """
        Get confidence in current phase classification (0-1).
        
        Returns: 1.0 if all recent frames agree, lower if unstable
        """
        if not self.phase_history or len(self.phase_history) < 5:
            return 0.5
        
        recent_phases = list(self.phase_history)[-5:]
        current_phase = self.get_phase()
        
        # What fraction agree with current phase?
        agreement = sum(1 for p in recent_phases if p == current_phase) / len(recent_phases)
        
        return agreement
    
    def is_weight_bearing(self):
        """Check if currently in any weight-bearing phase"""
        phase = self.get_phase()
        return phase in ["WEIGHT_BEARING_LEFT", "WEIGHT_BEARING_RIGHT"]
    
    def get_loaded_side(self):
        """
        Get which side is loaded (supporting weight).
        
        Returns: 'left', 'right', or None
        """
        phase = self.get_phase()
        
        if phase == "WEIGHT_BEARING_LEFT":
            return 'left'
        elif phase == "WEIGHT_BEARING_RIGHT":
            return 'right'
        else:
            return None