"""
Crutch Hole Scanner
Detects adjustment holes on crutches to measure:
- Distance between holes
- Number of adjustment positions
- Current overall length and handle height
"""

import cv2
import numpy as np


def scan_crutch_holes(side='right', pixels_per_cm=None, user_height_cm=175):
    """
    Scan crutch adjustment holes to measure crutch dimensions.

    The user shows the side of the crutch to the camera so the adjustment
    holes (small circles) are visible. The scanner detects:
    - Number of holes
    - Distance between holes
    - Total adjustable range

    Args:
        side: 'right' or 'left' crutch
        pixels_per_cm: Calibration factor (if None, will calibrate from user body)
        user_height_cm: User's height for body calibration

    Returns:
        dict with crutch configuration or None if failed/cancelled
    """

    print(f"\n{'='*70}")
    print(f"CRUTCH HOLE SCANNER - {side.upper()} CRUTCH")
    print(f"{'='*70}")

    # State machine
    STATE_CALIBRATING = 'calibrating'
    STATE_INSTRUCTIONS = 'instructions'
    STATE_POSITIONING = 'positioning'
    STATE_SCANNING = 'scanning'
    STATE_CONFIRM_OVERALL = 'confirm_overall'
    STATE_SCAN_HANDLE = 'scan_handle'
    STATE_CONFIRM_HANDLE = 'confirm_handle'
    STATE_COMPLETE = 'complete'

    # Determine if we need body calibration
    need_calibration = pixels_per_cm is None
    state = STATE_CALIBRATING if need_calibration else STATE_INSTRUCTIONS

    if need_calibration:
        print("\nFirst, we need to calibrate the camera distance.")
        print("Stand in front of the camera with your full body visible.")

    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return None

    # MediaPipe for body calibration (only if needed)
    mp_pose = None
    pose = None
    if need_calibration:
        try:
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=0
            )
        except ImportError:
            print("WARNING: MediaPipe not available, using default calibration")
            expected_leg_cm = user_height_cm * 0.53
            pixels_per_cm = 2.5  # Rough default
            state = STATE_INSTRUCTIONS

    # Tracking variables
    calibration_frames = []
    hole_detections = []  # List of detected hole patterns
    confirmed_holes = None
    overall_length_cm = None
    handle_height_cm = None
    handle_detections = []

    # Instructions shown flag
    instructions_shown_time = None

    print("\nControls:")
    print("  SPACE - Confirm/Continue")
    print("  R     - Retry detection")
    print("  Q     - Quit scanner")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        image = frame.copy()
        h, w = image.shape[:2]

        # ========================================
        # STATE: Calibrating (body reference)
        # ========================================
        if state == STATE_CALIBRATING:
            # Convert for MediaPipe
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Get hip and ankle positions
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

                # Calculate leg length in pixels
                lh_y = left_hip.y * h
                rh_y = right_hip.y * h
                la_y = left_ankle.y * h
                ra_y = right_ankle.y * h

                left_leg = abs(la_y - lh_y)
                right_leg = abs(ra_y - rh_y)
                avg_leg = (left_leg + right_leg) / 2

                calibration_frames.append(avg_leg)

                # Draw skeleton feedback
                import mediapipe as mp
                mp.solutions.drawing_utils.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Status display
                cv2.putText(image, f"Calibrating: {len(calibration_frames)}/30",
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(image, "Stand still, full body visible",
                           (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if len(calibration_frames) >= 30:
                    avg_pixels = np.mean(calibration_frames)
                    expected_leg_cm = user_height_cm * 0.53
                    pixels_per_cm = avg_pixels / expected_leg_cm
                    print(f"\nCalibration complete: {pixels_per_cm:.2f} pixels/cm")
                    state = STATE_INSTRUCTIONS
                    instructions_shown_time = cv2.getTickCount()
            else:
                cv2.putText(image, "Step back - full body must be visible",
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ========================================
        # STATE: Instructions
        # ========================================
        elif state == STATE_INSTRUCTIONS:
            # Dark overlay
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)

            # Instructions text
            lines = [
                "CRUTCH SCANNING INSTRUCTIONS",
                "",
                "1. Hold crutch SIDEWAYS to camera",
                "   (so adjustment holes are visible)",
                "",
                "2. Make sure the ENTIRE crutch length",
                "   fits in the frame",
                "",
                "3. Good lighting helps detection",
                "",
                "4. We'll scan the main tube first,",
                "   then the handle section",
                "",
                "Press SPACE when ready..."
            ]

            y = 60
            for i, line in enumerate(lines):
                color = (0, 255, 255) if i == 0 else (255, 255, 255)
                size = 0.8 if i == 0 else 0.6
                cv2.putText(image, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX,
                           size, color, 2 if i == 0 else 1)
                y += 35 if i == 0 else 30

        # ========================================
        # STATE: Positioning (waiting for crutch)
        # ========================================
        elif state == STATE_POSITIONING:
            # Draw guide area
            guide_top = 80
            guide_bottom = h - 80
            guide_left = w // 4
            guide_right = 3 * w // 4

            cv2.rectangle(image, (guide_left, guide_top), (guide_right, guide_bottom),
                         (0, 255, 0), 2)

            cv2.putText(image, "Position crutch in green box",
                       (w//2 - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, "Show the SIDE with adjustment holes",
                       (w//2 - 180, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, "Press SPACE to scan",
                       (w//2 - 100, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Try to detect holes in real-time for preview
            preview_holes = detect_holes(image[guide_top:guide_bottom, guide_left:guide_right])
            if preview_holes is not None and len(preview_holes) > 0:
                # Draw detected holes on preview
                for (x, y_hole, r) in preview_holes:
                    cv2.circle(image, (guide_left + x, guide_top + y_hole), r, (255, 0, 255), 2)
                cv2.putText(image, f"Detecting {len(preview_holes)} holes...",
                           (w//2 - 80, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # ========================================
        # STATE: Scanning (collecting hole data)
        # ========================================
        elif state == STATE_SCANNING:
            # Scan region
            scan_top = 50
            scan_bottom = h - 50
            scan_left = w // 6
            scan_right = 5 * w // 6

            cv2.rectangle(image, (scan_left, scan_top), (scan_right, scan_bottom),
                         (0, 255, 255), 2)

            # Detect holes
            roi = frame[scan_top:scan_bottom, scan_left:scan_right]
            holes = detect_holes(roi)

            if holes is not None and len(holes) >= 2:
                # Store detection
                hole_detections.append(holes)

                # Draw detected holes
                for (x, y_hole, r) in holes:
                    cv2.circle(image, (scan_left + x, scan_top + y_hole), r, (0, 255, 0), 2)
                    cv2.circle(image, (scan_left + x, scan_top + y_hole), 2, (0, 255, 0), -1)

                cv2.putText(image, f"Detected {len(holes)} holes - Frame {len(hole_detections)}/20",
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # After enough frames, analyze
                if len(hole_detections) >= 20:
                    confirmed_holes = analyze_hole_pattern(hole_detections, pixels_per_cm)
                    if confirmed_holes:
                        state = STATE_CONFIRM_OVERALL
                    else:
                        print("\nCould not get consistent hole pattern. Please retry.")
                        hole_detections = []
            else:
                cv2.putText(image, "Scanning... hold crutch steady",
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(image, "Make sure holes are visible",
                           (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ========================================
        # STATE: Confirm overall measurement
        # ========================================
        elif state == STATE_CONFIRM_OVERALL:
            # Display results
            overlay = image.copy()
            cv2.rectangle(overlay, (20, 20), (w - 20, 200), (0, 0, 0), -1)
            image = cv2.addWeighted(overlay, 0.8, image, 0.2, 0)

            cv2.putText(image, "OVERALL TUBE SCAN COMPLETE",
                       (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(image, f"Holes detected: {confirmed_holes['hole_count']}",
                       (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, f"Hole spacing: {confirmed_holes['hole_spacing_cm']:.2f} cm ({confirmed_holes['hole_spacing_cm']/2.54:.2f} in)",
                       (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, f"Adjustable range: {confirmed_holes['adjustable_range_cm']:.1f} cm",
                       (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(image, "Press SPACE to scan handle section",
                       (40, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, "Press R to retry overall scan",
                       (40, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # ========================================
        # STATE: Scan handle section
        # ========================================
        elif state == STATE_SCAN_HANDLE:
            cv2.putText(image, "Now show the HANDLE adjustment holes",
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(image, "(the holes where the hand grip attaches)",
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Scan region
            scan_top = 80
            scan_bottom = h - 80
            scan_left = w // 5
            scan_right = 4 * w // 5

            cv2.rectangle(image, (scan_left, scan_top), (scan_right, scan_bottom),
                         (255, 165, 0), 2)

            # Detect holes
            roi = frame[scan_top:scan_bottom, scan_left:scan_right]
            holes = detect_holes(roi)

            if holes is not None and len(holes) >= 2:
                handle_detections.append(holes)

                for (x, y_hole, r) in holes:
                    cv2.circle(image, (scan_left + x, scan_top + y_hole), r, (255, 165, 0), 2)

                cv2.putText(image, f"Detected {len(holes)} holes - Frame {len(handle_detections)}/20",
                           (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

                if len(handle_detections) >= 20:
                    handle_result = analyze_hole_pattern(handle_detections, pixels_per_cm)
                    if handle_result:
                        confirmed_holes['handle_holes'] = handle_result['hole_count']
                        confirmed_holes['handle_spacing_cm'] = handle_result['hole_spacing_cm']
                        state = STATE_CONFIRM_HANDLE
                    else:
                        print("\nCould not detect handle holes consistently. Retry or skip.")
                        handle_detections = []
            else:
                cv2.putText(image, "Position handle holes in orange box",
                           (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ========================================
        # STATE: Confirm handle measurement
        # ========================================
        elif state == STATE_CONFIRM_HANDLE:
            overlay = image.copy()
            cv2.rectangle(overlay, (20, 20), (w - 20, 220), (0, 0, 0), -1)
            image = cv2.addWeighted(overlay, 0.8, image, 0.2, 0)

            cv2.putText(image, "HANDLE SCAN COMPLETE",
                       (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

            cv2.putText(image, f"Handle holes: {confirmed_holes.get('handle_holes', 'N/A')}",
                       (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, f"Handle spacing: {confirmed_holes.get('handle_spacing_cm', 0):.2f} cm",
                       (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(image, "Enter current settings:",
                       (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(image, "Press SPACE to enter measurements",
                       (40, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ========================================
        # Display and handle input
        # ========================================
        cv2.imshow("Crutch Hole Scanner", image)

        key = cv2.waitKey(10) & 0xFF

        if key == ord('q') or key == 27:
            # Quit
            cap.release()
            cv2.destroyAllWindows()
            if pose:
                pose.close()
            return None

        elif key == ord(' '):
            if state == STATE_INSTRUCTIONS:
                state = STATE_POSITIONING
            elif state == STATE_POSITIONING:
                state = STATE_SCANNING
                hole_detections = []
            elif state == STATE_CONFIRM_OVERALL:
                state = STATE_SCAN_HANDLE
                handle_detections = []
            elif state == STATE_CONFIRM_HANDLE:
                state = STATE_COMPLETE

        elif key == ord('r') or key == ord('R'):
            if state == STATE_CONFIRM_OVERALL:
                state = STATE_POSITIONING
                hole_detections = []
                confirmed_holes = None
            elif state in [STATE_SCAN_HANDLE, STATE_CONFIRM_HANDLE]:
                state = STATE_SCAN_HANDLE
                handle_detections = []

        elif key == ord('s') or key == ord('S'):
            # Skip handle scan
            if state == STATE_SCAN_HANDLE:
                confirmed_holes['handle_holes'] = None
                confirmed_holes['handle_spacing_cm'] = None
                state = STATE_CONFIRM_HANDLE

        if state == STATE_COMPLETE:
            break

    cap.release()
    cv2.destroyAllWindows()
    if pose:
        pose.close()

    # ========================================
    # Get current settings from user
    # ========================================
    if confirmed_holes:
        print(f"\n{'='*70}")
        print("CRUTCH SCAN RESULTS")
        print(f"{'='*70}")
        print(f"\nOverall tube:")
        print(f"  Adjustment holes: {confirmed_holes['hole_count']}")
        print(f"  Hole spacing: {confirmed_holes['hole_spacing_cm']:.2f} cm ({confirmed_holes['hole_spacing_cm']/2.54:.2f} inches)")
        print(f"  Adjustable range: {confirmed_holes['adjustable_range_cm']:.1f} cm")

        if confirmed_holes.get('handle_holes'):
            print(f"\nHandle tube:")
            print(f"  Adjustment holes: {confirmed_holes['handle_holes']}")
            print(f"  Hole spacing: {confirmed_holes['handle_spacing_cm']:.2f} cm")

        print(f"\nNow enter your CURRENT crutch settings:")

        try:
            overall_input = input(f"Current overall length in cm (or press Enter to estimate): ").strip()
            if overall_input:
                overall_length_cm = float(overall_input)
            else:
                # Estimate from typical crutch
                overall_length_cm = 120.0  # Default mid-range
                print(f"  Using default estimate: {overall_length_cm} cm")

            handle_input = input(f"Current handle height in cm (or press Enter to estimate): ").strip()
            if handle_input:
                handle_height_cm = float(handle_input)
            else:
                handle_height_cm = user_height_cm * 0.485  # Wrist height estimate
                print(f"  Using estimate based on your height: {handle_height_cm:.1f} cm")

        except ValueError:
            print("Invalid input, using estimates")
            overall_length_cm = 120.0
            handle_height_cm = user_height_cm * 0.485

        return {
            'side': side,
            'length_cm': overall_length_cm,
            'handle_height_cm': handle_height_cm,
            'hole_count': confirmed_holes['hole_count'],
            'hole_spacing_cm': confirmed_holes['hole_spacing_cm'],
            'adjustable_range_cm': confirmed_holes['adjustable_range_cm'],
            'handle_holes': confirmed_holes.get('handle_holes'),
            'handle_spacing_cm': confirmed_holes.get('handle_spacing_cm'),
            'pixels_per_cm': pixels_per_cm
        }

    return None


def detect_holes(roi):
    """
    Detect circular holes in a region of interest.

    Args:
        roi: Image region to analyze (BGR format)

    Returns:
        numpy array of (x, y, radius) for each detected circle, or None
    """
    if roi is None or roi.size == 0:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=15,  # Min distance between circle centers
        param1=50,   # Upper threshold for Canny edge detector
        param2=30,   # Accumulator threshold for circle detection
        minRadius=3,  # Min radius
        maxRadius=25  # Max radius
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0]  # Return array of (x, y, r)

    return None


def analyze_hole_pattern(detections, pixels_per_cm):
    """
    Analyze multiple frame detections to get consistent hole pattern.

    Args:
        detections: List of hole detection arrays
        pixels_per_cm: Calibration factor

    Returns:
        dict with hole_count, hole_spacing_cm, adjustable_range_cm or None
    """
    if not detections or len(detections) < 5:
        return None

    # Count holes in each detection
    hole_counts = [len(d) for d in detections]

    # Use median hole count (most robust)
    median_count = int(np.median(hole_counts))

    if median_count < 2:
        return None

    # Filter detections with the median count
    valid_detections = [d for d in detections if len(d) == median_count]

    if len(valid_detections) < 5:
        # Fall back to detections close to median
        valid_detections = [d for d in detections if abs(len(d) - median_count) <= 1]

    if len(valid_detections) < 3:
        return None

    # Calculate average spacing between holes
    spacings = []
    for detection in valid_detections:
        # Sort holes by y-coordinate (assuming vertical crutch)
        sorted_holes = sorted(detection, key=lambda h: h[1])

        # Calculate distances between adjacent holes
        for i in range(len(sorted_holes) - 1):
            y1 = sorted_holes[i][1]
            y2 = sorted_holes[i + 1][1]
            spacing_px = abs(y2 - y1)
            spacings.append(spacing_px)

    if not spacings:
        return None

    # Get median spacing (robust to outliers)
    median_spacing_px = np.median(spacings)

    # Filter out extreme outliers
    filtered_spacings = [s for s in spacings if 0.5 * median_spacing_px < s < 1.5 * median_spacing_px]

    if not filtered_spacings:
        filtered_spacings = spacings

    avg_spacing_px = np.mean(filtered_spacings)
    spacing_cm = avg_spacing_px / pixels_per_cm

    # Total adjustable range
    adjustable_range_cm = spacing_cm * (median_count - 1)

    return {
        'hole_count': median_count,
        'hole_spacing_cm': spacing_cm,
        'adjustable_range_cm': adjustable_range_cm
    }


print("Crutch hole scanner loaded")
