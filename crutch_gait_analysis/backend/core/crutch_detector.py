"""
Crutch Detection Module

Detects forearm crutches in video frames using computer vision techniques.
Uses a heuristic approach based on:
1. Wrist positions as anchor points (from MediaPipe)
2. Edge detection to find crutch shafts
3. Line detection to track crutch orientation
4. Floor estimation for tip position

This enables measurement of:
- Crutch tip placement (distance from body)
- Crutch symmetry (left vs right placement)
- Crutch angle (vertical alignment)
"""

import cv2
import numpy as np
from collections import deque


class CrutchDetector:
    """
    Detects and tracks forearm crutches using computer vision.

    Uses wrist positions from MediaPipe as anchor points, then applies
    edge detection and line detection to find crutch shafts.
    """

    def __init__(self, history_size=10):
        """
        Initialize crutch detector.

        Args:
            history_size: Number of frames to smooth detections over
        """
        self.history_size = history_size

        # Detection history for smoothing
        self.left_tip_history = deque(maxlen=history_size)
        self.right_tip_history = deque(maxlen=history_size)
        self.left_angle_history = deque(maxlen=history_size)
        self.right_angle_history = deque(maxlen=history_size)

        # Canny edge detection parameters
        self.canny_low = 50
        self.canny_high = 150

        # Hough line parameters
        self.hough_threshold = 30
        self.min_line_length = 50
        self.max_line_gap = 20

        # Search region parameters (relative to wrist position)
        self.search_width = 80  # pixels left/right of wrist
        self.search_height_above = 50  # pixels above wrist
        self.search_height_below = 300  # pixels below wrist (toward floor)

        # Expected crutch angle range (degrees from vertical)
        self.min_angle = -30  # tilted left
        self.max_angle = 30   # tilted right

    def detect(self, frame, landmarks, image_width, image_height,
               floor_y=None, pixels_per_cm=None):
        """
        Detect crutches in the frame.

        Args:
            frame: BGR image from camera
            landmarks: MediaPipe pose landmarks
            image_width: Frame width
            image_height: Frame height
            floor_y: Y-coordinate of floor (if known from calibration)
            pixels_per_cm: Scale factor (if calibrated)

        Returns:
            dict with detection results:
                - left_tip: (x, y) of left crutch tip or None
                - right_tip: (x, y) of right crutch tip or None
                - left_angle: Angle from vertical (degrees) or None
                - right_angle: Angle from vertical (degrees) or None
                - tip_distance_px: Distance between tips in pixels
                - tip_symmetry: Ratio of left/right distance from body center
                - confidence: Detection confidence (0-1)
        """
        # Get wrist positions
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28

        left_wrist = landmarks[LEFT_WRIST]
        right_wrist = landmarks[RIGHT_WRIST]

        # Convert to pixel coordinates
        lw_x = int(left_wrist.x * image_width)
        lw_y = int(left_wrist.y * image_height)
        rw_x = int(right_wrist.x * image_width)
        rw_y = int(right_wrist.y * image_height)

        # Estimate floor level if not provided
        if floor_y is None:
            left_ankle = landmarks[LEFT_ANKLE]
            right_ankle = landmarks[RIGHT_ANKLE]
            floor_y = int(max(left_ankle.y, right_ankle.y) * image_height) + 20

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect left crutch
        left_result = self._detect_single_crutch(
            gray, lw_x, lw_y, floor_y, image_width, image_height, side='left'
        )

        # Detect right crutch
        right_result = self._detect_single_crutch(
            gray, rw_x, rw_y, floor_y, image_width, image_height, side='right'
        )

        # Update history
        if left_result['tip'] is not None:
            self.left_tip_history.append(left_result['tip'])
            self.left_angle_history.append(left_result['angle'])

        if right_result['tip'] is not None:
            self.right_tip_history.append(right_result['tip'])
            self.right_angle_history.append(right_result['angle'])

        # Get smoothed values
        left_tip = self._get_smoothed_tip(self.left_tip_history)
        right_tip = self._get_smoothed_tip(self.right_tip_history)
        left_angle = self._get_smoothed_angle(self.left_angle_history)
        right_angle = self._get_smoothed_angle(self.right_angle_history)

        # Calculate metrics
        tip_distance_px = None
        tip_symmetry = None
        body_center_x = (lw_x + rw_x) / 2

        if left_tip is not None and right_tip is not None:
            tip_distance_px = np.sqrt(
                (right_tip[0] - left_tip[0])**2 +
                (right_tip[1] - left_tip[1])**2
            )

            # Symmetry: ratio of distances from body center
            left_dist = abs(left_tip[0] - body_center_x)
            right_dist = abs(right_tip[0] - body_center_x)

            if right_dist > 0:
                tip_symmetry = left_dist / right_dist
            else:
                tip_symmetry = 1.0 if left_dist == 0 else float('inf')

        # Calculate confidence based on detection consistency
        left_conf = len(self.left_tip_history) / self.history_size
        right_conf = len(self.right_tip_history) / self.history_size
        confidence = (left_conf + right_conf) / 2

        return {
            'left_tip': left_tip,
            'right_tip': right_tip,
            'left_angle': left_angle,
            'right_angle': right_angle,
            'tip_distance_px': tip_distance_px,
            'tip_symmetry': tip_symmetry,
            'confidence': confidence,
            'floor_y': floor_y
        }

    def _detect_single_crutch(self, gray, wrist_x, wrist_y, floor_y,
                               image_width, image_height, side='left'):
        """
        Detect a single crutch near a wrist position.

        Uses edge detection and Hough line transform to find
        vertical-ish lines extending from the wrist toward the floor.
        """
        # Define search region
        if side == 'left':
            x1 = max(0, wrist_x - self.search_width)
            x2 = wrist_x + self.search_width // 2
        else:
            x1 = wrist_x - self.search_width // 2
            x2 = min(image_width, wrist_x + self.search_width)

        y1 = max(0, wrist_y - self.search_height_above)
        y2 = min(image_height, min(floor_y + 30, wrist_y + self.search_height_below))

        # Extract region of interest
        roi = gray[y1:y2, x1:x2]

        if roi.size == 0:
            return {'tip': None, 'angle': None, 'lines': []}

        # Apply edge detection
        edges = cv2.Canny(roi, self.canny_low, self.canny_high)

        # Apply morphological operations to connect edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        if lines is None:
            return {'tip': None, 'angle': None, 'lines': []}

        # Filter lines by angle (should be roughly vertical)
        valid_lines = []
        for line in lines:
            x1_l, y1_l, x2_l, y2_l = line[0]

            # Calculate angle from vertical
            if y2_l != y1_l:
                angle = np.degrees(np.arctan2(x2_l - x1_l, y2_l - y1_l))
            else:
                angle = 90  # Horizontal line

            # Keep lines that are roughly vertical
            if self.min_angle <= angle <= self.max_angle:
                # Convert back to full frame coordinates
                full_x1 = x1_l + x1
                full_y1 = y1_l + y1
                full_x2 = x2_l + x1
                full_y2 = y2_l + y1

                # Ensure y2 > y1 (line goes downward)
                if full_y1 > full_y2:
                    full_x1, full_y1, full_x2, full_y2 = full_x2, full_y2, full_x1, full_y1

                valid_lines.append({
                    'x1': full_x1, 'y1': full_y1,
                    'x2': full_x2, 'y2': full_y2,
                    'angle': angle,
                    'length': np.sqrt((x2_l - x1_l)**2 + (y2_l - y1_l)**2)
                })

        if not valid_lines:
            return {'tip': None, 'angle': None, 'lines': []}

        # Find the longest valid line (most likely to be the crutch shaft)
        best_line = max(valid_lines, key=lambda l: l['length'])

        # Extrapolate to find the tip (where line meets floor)
        if best_line['y2'] != best_line['y1']:
            slope = (best_line['x2'] - best_line['x1']) / (best_line['y2'] - best_line['y1'])
            tip_x = best_line['x2'] + slope * (floor_y - best_line['y2'])
            tip_y = floor_y
        else:
            tip_x = best_line['x2']
            tip_y = floor_y

        # Clamp tip position to valid range
        tip_x = max(0, min(image_width - 1, int(tip_x)))
        tip_y = max(0, min(image_height - 1, int(tip_y)))

        return {
            'tip': (tip_x, tip_y),
            'angle': best_line['angle'],
            'lines': valid_lines
        }

    def _get_smoothed_tip(self, history):
        """Get smoothed tip position from history."""
        if len(history) < 3:
            return None

        tips = list(history)
        avg_x = np.mean([t[0] for t in tips])
        avg_y = np.mean([t[1] for t in tips])
        return (int(avg_x), int(avg_y))

    def _get_smoothed_angle(self, history):
        """Get smoothed angle from history."""
        if len(history) < 3:
            return None
        return np.mean(list(history))

    def draw_detections(self, frame, detections, show_search_regions=False):
        """
        Draw crutch detections on the frame.

        Args:
            frame: BGR image to draw on
            detections: Result dict from detect()
            show_search_regions: Whether to show debug info

        Returns:
            Frame with detections drawn
        """
        # Draw left crutch tip
        if detections['left_tip'] is not None:
            cv2.circle(frame, detections['left_tip'], 8, (255, 0, 0), -1)
            cv2.putText(frame, "L",
                       (detections['left_tip'][0] - 5, detections['left_tip'][1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw right crutch tip
        if detections['right_tip'] is not None:
            cv2.circle(frame, detections['right_tip'], 8, (0, 0, 255), -1)
            cv2.putText(frame, "R",
                       (detections['right_tip'][0] - 5, detections['right_tip'][1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw line between tips if both detected
        if detections['left_tip'] is not None and detections['right_tip'] is not None:
            cv2.line(frame, detections['left_tip'], detections['right_tip'],
                    (0, 255, 255), 2)

            # Show distance
            mid_x = (detections['left_tip'][0] + detections['right_tip'][0]) // 2
            mid_y = (detections['left_tip'][1] + detections['right_tip'][1]) // 2

            if detections['tip_distance_px'] is not None:
                cv2.putText(frame, f"{detections['tip_distance_px']:.0f}px",
                           (mid_x - 20, mid_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Draw floor line
        if detections['floor_y'] is not None:
            h, w = frame.shape[:2]
            cv2.line(frame, (0, detections['floor_y']), (w, detections['floor_y']),
                    (100, 100, 100), 1)

        return frame

    def evaluate_placement(self, detections, hip_width_px, pixels_per_cm=None):
        """
        Evaluate crutch tip placement quality.

        Args:
            detections: Result dict from detect()
            hip_width_px: Hip width in pixels (for normalization)
            pixels_per_cm: Scale factor if available

        Returns:
            dict with evaluation results:
                - placement_status: 'good', 'warning', 'critical', or 'unknown'
                - symmetry_status: 'good', 'warning', 'critical', or 'unknown'
                - angle_status: 'good', 'warning', 'critical', or 'unknown'
                - messages: List of feedback messages
        """
        messages = []

        # Check if crutches detected
        if detections['left_tip'] is None and detections['right_tip'] is None:
            return {
                'placement_status': 'unknown',
                'symmetry_status': 'unknown',
                'angle_status': 'unknown',
                'messages': ["Crutches not detected"]
            }

        # ========================================
        # Evaluate symmetry (left vs right distance from center)
        # ========================================
        symmetry_status = 'unknown'
        if detections['tip_symmetry'] is not None:
            sym = detections['tip_symmetry']

            if 0.85 <= sym <= 1.15:  # Within 15%
                symmetry_status = 'good'
            elif 0.7 <= sym <= 1.3:  # Within 30%
                symmetry_status = 'warning'
                if sym < 1:
                    messages.append("Left crutch placed closer than right")
                else:
                    messages.append("Right crutch placed closer than left")
            else:
                symmetry_status = 'critical'
                messages.append("Crutch placement very asymmetric")

        # ========================================
        # Evaluate tip spread (distance between tips)
        # ========================================
        placement_status = 'unknown'
        if detections['tip_distance_px'] is not None and hip_width_px > 0:
            # Normalize by hip width
            spread_ratio = detections['tip_distance_px'] / hip_width_px

            # Optimal: tips should be ~1.5-2.5x hip width apart
            if 1.3 <= spread_ratio <= 2.8:
                placement_status = 'good'
            elif 1.0 <= spread_ratio <= 3.2:
                placement_status = 'warning'
                if spread_ratio < 1.3:
                    messages.append("Crutch tips too close together")
                else:
                    messages.append("Crutch tips very wide apart")
            else:
                placement_status = 'critical'
                if spread_ratio < 1.0:
                    messages.append("Crutch tips dangerously close")
                else:
                    messages.append("Crutch tips extremely wide - stability risk")

        # ========================================
        # Evaluate crutch angles
        # ========================================
        angle_status = 'unknown'
        angles = []
        if detections['left_angle'] is not None:
            angles.append(('left', detections['left_angle']))
        if detections['right_angle'] is not None:
            angles.append(('right', detections['right_angle']))

        if angles:
            max_angle = max(abs(a[1]) for a in angles)

            if max_angle <= 10:
                angle_status = 'good'
            elif max_angle <= 20:
                angle_status = 'warning'
                for side, angle in angles:
                    if abs(angle) > 10:
                        direction = "outward" if angle > 0 else "inward"
                        messages.append(f"{side.capitalize()} crutch tilted {direction}")
            else:
                angle_status = 'critical'
                messages.append("Crutch angle excessive - check grip position")

        return {
            'placement_status': placement_status,
            'symmetry_status': symmetry_status,
            'angle_status': angle_status,
            'messages': messages
        }

    def reset(self):
        """Reset detection history."""
        self.left_tip_history.clear()
        self.right_tip_history.clear()
        self.left_angle_history.clear()
        self.right_angle_history.clear()


# Convenience function for quick testing
if __name__ == '__main__':
    print("Crutch Detector - Test Mode")
    print("This module detects forearm crutches using computer vision.")
    print("It requires a video frame and MediaPipe pose landmarks to work.")
