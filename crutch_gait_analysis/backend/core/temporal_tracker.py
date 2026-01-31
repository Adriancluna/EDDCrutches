"""
Temporal tracking and smoothing
"""
import numpy as np
from collections import deque


class TemporalTracker:
    """Track measurements over time windows to reduce noise"""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        
        self.history = {
            'elbow_r': deque(maxlen=window_size),
            'elbow_l': deque(maxlen=window_size),
            'trunk_lean': deque(maxlen=window_size),
            'knee_r': deque(maxlen=window_size),
            'knee_l': deque(maxlen=window_size),
            'step_length': deque(maxlen=window_size),
            'base_width': deque(maxlen=window_size),
            'shoulder_asym': deque(maxlen=window_size)
        }
        
        self.status_history = {
            'elbow_r': deque(maxlen=window_size),
            'elbow_l': deque(maxlen=window_size),
            'trunk': deque(maxlen=window_size),
            'knee_r': deque(maxlen=window_size),
            'knee_l': deque(maxlen=window_size),
            'step': deque(maxlen=window_size),
            'base': deque(maxlen=window_size),
            'shoulder': deque(maxlen=window_size)
        }
    
    def update(self, measurements, evaluations):
        """Add new measurements and evaluations to history"""
        for key, value in measurements.items():
            if key in self.history:
                self.history[key].append(value)
        
        for key, eval_result in evaluations.items():
            if key in self.status_history:
                self.status_history[key].append(eval_result['status'])
    
    def get_smoothed(self, metric):
        """Get moving average of a measurement"""
        if metric not in self.history or len(self.history[metric]) == 0:
            return None
        return np.mean(list(self.history[metric]))
    
    def is_issue_persistent(self, metric, min_persistence=0.6):
        """Check if an issue persists across the time window"""
        if metric not in self.status_history:
            return False
        
        statuses = list(self.status_history[metric])
        
        if len(statuses) < self.window_size // 2:
            return False
        
        problem_count = sum(1 for status in statuses 
                           if status in ['warning', 'critical'])
        
        persistence = problem_count / len(statuses)
        
        return persistence >= min_persistence