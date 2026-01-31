"""
Geometric calculation utilities for pose analysis.

Supports both 2D (pixel) and 3D (world) coordinate systems.

For angles: Use 3D world landmarks for accuracy (camera angle independent)
For distances: Use 2D pixel landmarks with calibration (actual floor distances)
"""
import math

SQRT = math.sqrt
ACOS = math.acos
DEGREES = math.degrees

# Visibility threshold for landmark confidence
DEFAULT_VISIBILITY_THRESHOLD = 0.5


def compute_distance(point_a, point_b):
    """
    Fast 2D distance calculation.

    Args:
        point_a: (x, y) tuple
        point_b: (x, y) tuple

    Returns:
        Euclidean distance between points
    """
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return SQRT(dx * dx + dy * dy)


def compute_distance_3d(point_a, point_b):
    """
    Fast 3D distance calculation.

    Args:
        point_a: (x, y, z) tuple
        point_b: (x, y, z) tuple

    Returns:
        Euclidean distance between points in 3D space
    """
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    dz = point_a[2] - point_b[2]
    return SQRT(dx * dx + dy * dy + dz * dz)


def compute_angle(point_a, point_b, point_c):
    """
    Fast 2D angle calculation at point_b.

    Calculates the angle ABC where B is the vertex.

    Args:
        point_a: (x, y) first point
        point_b: (x, y) vertex point
        point_c: (x, y) third point

    Returns:
        Angle in degrees (0-180)
    """
    ax, ay = point_a
    bx, by = point_b
    cx, cy = point_c

    ba_x = ax - bx
    ba_y = ay - by
    bc_x = cx - bx
    bc_y = cy - by

    dot_product = ba_x * bc_x + ba_y * bc_y
    mag_ba = SQRT(ba_x * ba_x + ba_y * ba_y)
    mag_bc = SQRT(bc_x * bc_x + bc_y * bc_y)

    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return 0.0

    cos_angle = max(-1.0, min(1.0, dot_product / (mag_ba * mag_bc)))
    return DEGREES(ACOS(cos_angle))


def compute_angle_3d(point_a, point_b, point_c):
    """
    3D angle calculation at point_b.

    This is camera-angle independent and more accurate for joint angles.

    Args:
        point_a: (x, y, z) first point
        point_b: (x, y, z) vertex point
        point_c: (x, y, z) third point

    Returns:
        Angle in degrees (0-180)
    """
    ax, ay, az = point_a
    bx, by, bz = point_b
    cx, cy, cz = point_c

    # Vectors from B to A and B to C
    ba_x = ax - bx
    ba_y = ay - by
    ba_z = az - bz
    bc_x = cx - bx
    bc_y = cy - by
    bc_z = cz - bz

    # Dot product
    dot_product = ba_x * bc_x + ba_y * bc_y + ba_z * bc_z

    # Magnitudes
    mag_ba = SQRT(ba_x * ba_x + ba_y * ba_y + ba_z * ba_z)
    mag_bc = SQRT(bc_x * bc_x + bc_y * bc_y + bc_z * bc_z)

    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return 0.0

    cos_angle = max(-1.0, min(1.0, dot_product / (mag_ba * mag_bc)))
    return DEGREES(ACOS(cos_angle))


def compute_angle_from_vertical(point_a, point_b):
    """
    Fast 2D vertical angle calculation.

    Calculates angle between vector (A-B) and vertical (0, -1).

    Args:
        point_a: (x, y) upper point
        point_b: (x, y) lower point

    Returns:
        Angle from vertical in degrees (0 = perfectly upright)
    """
    vec_x = point_a[0] - point_b[0]
    vec_y = point_a[1] - point_b[1]

    # Vertical vector is (0, -1) in image coordinates
    dot_product = -vec_y
    mag_vec = SQRT(vec_x * vec_x + vec_y * vec_y)

    if mag_vec < 1e-6:
        return 0.0

    cos_angle = max(-1.0, min(1.0, dot_product / mag_vec))
    return DEGREES(ACOS(cos_angle))


def compute_angle_from_vertical_3d(point_a, point_b):
    """
    3D vertical angle calculation (trunk lean in world coordinates).

    Uses Y-axis as vertical in MediaPipe world coordinates.

    Args:
        point_a: (x, y, z) upper point (shoulders)
        point_b: (x, y, z) lower point (hips)

    Returns:
        Angle from vertical in degrees
    """
    vec_x = point_a[0] - point_b[0]
    vec_y = point_a[1] - point_b[1]
    vec_z = point_a[2] - point_b[2]

    # Vertical vector in world coords is (0, -1, 0)
    # (Y points down in MediaPipe world landmarks)
    dot_product = -vec_y
    mag_vec = SQRT(vec_x * vec_x + vec_y * vec_y + vec_z * vec_z)

    if mag_vec < 1e-6:
        return 0.0

    cos_angle = max(-1.0, min(1.0, dot_product / mag_vec))
    return DEGREES(ACOS(cos_angle))


def check_visibility(landmarks, indices, threshold=DEFAULT_VISIBILITY_THRESHOLD):
    """
    Check if all specified landmarks have sufficient visibility.

    Args:
        landmarks: MediaPipe landmark list
        indices: List of landmark indices to check
        threshold: Minimum visibility score (0-1)

    Returns:
        True if all landmarks are visible above threshold
    """
    for idx in indices:
        if landmarks[idx].visibility < threshold:
            return False
    return True


def get_visibility_mask(landmarks, threshold=DEFAULT_VISIBILITY_THRESHOLD):
    """
    Get visibility status for all landmarks.

    Args:
        landmarks: MediaPipe landmark list
        threshold: Minimum visibility score

    Returns:
        Dict mapping landmark index to visibility boolean
    """
    return {
        i: lm.visibility >= threshold
        for i, lm in enumerate(landmarks)
    }


def extract_point_2d(landmark, image_width, image_height):
    """
    Extract 2D pixel coordinates from a landmark.

    Args:
        landmark: MediaPipe landmark
        image_width: Frame width in pixels
        image_height: Frame height in pixels

    Returns:
        (x, y) tuple in pixel coordinates
    """
    return (
        landmark.x * image_width,
        landmark.y * image_height
    )


def extract_point_3d(landmark):
    """
    Extract 3D world coordinates from a landmark.

    MediaPipe world landmarks are in meters, centered at hip midpoint.

    Args:
        landmark: MediaPipe world landmark

    Returns:
        (x, y, z) tuple in world coordinates
    """
    return (landmark.x, landmark.y, landmark.z)


class PoseGeometry:
    """
    High-level interface for pose geometry calculations.

    Handles visibility gating and coordinate extraction automatically.
    Uses 3D landmarks for angles, 2D for distances.
    """

    def __init__(self, visibility_threshold=DEFAULT_VISIBILITY_THRESHOLD):
        self.visibility_threshold = visibility_threshold

    def calculate_elbow_angle(self, landmarks_2d, landmarks_3d, side='right',
                              use_3d=True):
        """
        Calculate elbow flexion angle.

        Args:
            landmarks_2d: MediaPipe pose_landmarks (2D)
            landmarks_3d: MediaPipe pose_world_landmarks (3D)
            side: 'right' or 'left'
            use_3d: Use 3D landmarks for accuracy

        Returns:
            Angle in degrees, or None if landmarks not visible
        """
        if side == 'right':
            shoulder_idx, elbow_idx, wrist_idx = 12, 14, 16
        else:
            shoulder_idx, elbow_idx, wrist_idx = 11, 13, 15

        # Check visibility using 2D landmarks
        if not check_visibility(landmarks_2d.landmark,
                                [shoulder_idx, elbow_idx, wrist_idx],
                                self.visibility_threshold):
            return None

        if use_3d and landmarks_3d is not None:
            lm = landmarks_3d.landmark
            return compute_angle_3d(
                extract_point_3d(lm[shoulder_idx]),
                extract_point_3d(lm[elbow_idx]),
                extract_point_3d(lm[wrist_idx])
            )
        else:
            # Fallback to 2D if 3D not available
            lm = landmarks_2d.landmark
            # Note: This requires image dimensions to be passed
            return None

    def calculate_knee_angle(self, landmarks_2d, landmarks_3d, side='right',
                             use_3d=True):
        """
        Calculate knee flexion angle.

        Args:
            landmarks_2d: MediaPipe pose_landmarks (2D)
            landmarks_3d: MediaPipe pose_world_landmarks (3D)
            side: 'right' or 'left'
            use_3d: Use 3D landmarks for accuracy

        Returns:
            Angle in degrees, or None if landmarks not visible
        """
        if side == 'right':
            hip_idx, knee_idx, ankle_idx = 24, 26, 28
        else:
            hip_idx, knee_idx, ankle_idx = 23, 25, 27

        if not check_visibility(landmarks_2d.landmark,
                                [hip_idx, knee_idx, ankle_idx],
                                self.visibility_threshold):
            return None

        if use_3d and landmarks_3d is not None:
            lm = landmarks_3d.landmark
            return compute_angle_3d(
                extract_point_3d(lm[hip_idx]),
                extract_point_3d(lm[knee_idx]),
                extract_point_3d(lm[ankle_idx])
            )
        else:
            return None

    def calculate_trunk_lean(self, landmarks_2d, landmarks_3d, use_3d=True):
        """
        Calculate trunk forward lean angle.

        Args:
            landmarks_2d: MediaPipe pose_landmarks (2D)
            landmarks_3d: MediaPipe pose_world_landmarks (3D)
            use_3d: Use 3D landmarks for accuracy

        Returns:
            Angle from vertical in degrees, or None if not visible
        """
        # Indices for shoulders and hips
        left_shoulder, right_shoulder = 11, 12
        left_hip, right_hip = 23, 24

        if not check_visibility(landmarks_2d.landmark,
                                [left_shoulder, right_shoulder, left_hip, right_hip],
                                self.visibility_threshold):
            return None

        if use_3d and landmarks_3d is not None:
            lm = landmarks_3d.landmark
            # Mid-shoulder point
            mid_shoulder = (
                (lm[left_shoulder].x + lm[right_shoulder].x) / 2,
                (lm[left_shoulder].y + lm[right_shoulder].y) / 2,
                (lm[left_shoulder].z + lm[right_shoulder].z) / 2
            )
            # Mid-hip point
            mid_hip = (
                (lm[left_hip].x + lm[right_hip].x) / 2,
                (lm[left_hip].y + lm[right_hip].y) / 2,
                (lm[left_hip].z + lm[right_hip].z) / 2
            )
            return compute_angle_from_vertical_3d(mid_shoulder, mid_hip)
        else:
            return None
