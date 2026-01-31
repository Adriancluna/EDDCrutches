"""
Cell 5.5: Crutch Length Scanner
Measures crutch dimensions using camera calibration
"""

import cv2
import numpy as np
import mediapipe as mp

def scan_crutch_simple(crutch_name='right crutch', pixels_per_cm=None):
    """
    Scan crutch to measure overall length and handle height.
    
    Args:
        crutch_name: Name for display (e.g., 'right crutch')
        pixels_per_cm: Calibration factor (if None, will compute from body)
    
    Returns:
        dict with measurements or None if failed
    """
    
    print(f"\n{'='*70}")
    print(f"SCANNING: {crutch_name.upper()}")
    print(f"{'='*70}")
    
    print(f"\n📸 Setup Instructions:")
    print(f"   1. Place crutch VERTICALLY against a wall")
    print(f"   2. Stand at the SAME distance as your calibration")
    print(f"   3. Ensure ENTIRE crutch is visible (tip to top)")
    print(f"   4. Make sure lighting is good")
    
    # If no calibration provided, use body reference
    need_body_reference = pixels_per_cm is None
    
    if need_body_reference:
        print(f"\n👤 First, we need to calibrate distance...")
        print(f"   Stand in front of camera, arms at sides")
        input("   Press Enter when ready...")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    mp_pose = mp.solutions.pose
    
    # State machine
    STATE_CALIBRATING = 'calibrating'
    STATE_READY_FOR_CRUTCH = 'ready'
    STATE_SCANNING_CRUTCH = 'scanning'
    STATE_COMPLETE = 'complete'
    
    state = STATE_CALIBRATING if need_body_reference else STATE_READY_FOR_CRUTCH
    
    calibration_frames = []
    crutch_measurements = []
    frame_count = 0
    
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0) as pose:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            image_height, image_width = image.shape[:2]
            
            # ========================================
            # STATE: Calibrating (if needed)
            # ========================================
            if state == STATE_CALIBRATING:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Get hip and ankle
                    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                    
                    # Measure leg length in pixels
                    lx_hip = left_hip.x * image_width
                    ly_hip = left_hip.y * image_height
                    rx_hip = right_hip.x * image_width
                    ry_hip = right_hip.y * image_height
                    lx_ankle = left_ankle.x * image_width
                    ly_ankle = left_ankle.y * image_height
                    rx_ankle = right_ankle.x * image_width
                    ry_ankle = right_ankle.y * image_height
                    
                    left_leg = np.sqrt((lx_hip - lx_ankle)**2 + (ly_hip - ly_ankle)**2)
                    right_leg = np.sqrt((rx_hip - rx_ankle)**2 + (ry_hip - ry_ankle)**2)
                    avg_leg = (left_leg + right_leg) / 2
                    
                    calibration_frames.append(avg_leg)
                    
                    # Draw skeleton
                    mp.solutions.drawing_utils.draw_landmarks(
                        image, results.pose_landmarks, 
                        mp_pose.POSE_CONNECTIONS)
                    
                    # Status
                    cv2.putText(image, f"Calibrating: {len(calibration_frames)}/30", 
                               (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.8, (0, 255, 255), 2)
                    
                    if len(calibration_frames) >= 30:
                        avg_leg_pixels = np.mean(calibration_frames)
                        # Assume user height is 175cm, leg is 53% of height
                        expected_leg_cm = 175 * 0.53  # Adjust if you know user's actual height
                        pixels_per_cm = avg_leg_pixels / expected_leg_cm
                        
                        print(f"\n✓ Calibration complete: {pixels_per_cm:.2f} pixels/cm")
                        state = STATE_READY_FOR_CRUTCH
                
                else:
                    cv2.putText(image, "Stand in frame, arms at sides", 
                               (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 0, 255), 2)
            
            # ========================================
            # STATE: Ready for Crutch
            # ========================================
            elif state == STATE_READY_FOR_CRUTCH:
                cv2.putText(image, "Position crutch vertically", 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (0, 255, 0), 2)
                cv2.putText(image, "Press SPACE when ready", 
                           (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 255), 2)
                
                # Draw guide box
                box_left = image_width // 3
                box_right = 2 * image_width // 3
                cv2.rectangle(image, (box_left, 50), (box_right, image_height - 50),
                             (0, 255, 0), 2)
                cv2.putText(image, "Position crutch here", 
                           (box_left + 20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # ========================================
            # STATE: Scanning Crutch
            # ========================================
            elif state == STATE_SCANNING_CRUTCH:
                # Convert to grayscale for edge detection
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Edge detection
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)
                
                # Hough line detection (find long vertical lines)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                                       minLineLength=300, maxLineGap=20)
                
                if lines is not None:
                    # Find most vertical line (the crutch shaft)
                    best_line = None
                    max_length = 0
                    
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                        
                        # Check if vertical (angle close to 90 degrees)
                        if x2 - x1 != 0:
                            angle = np.abs(np.arctan2(y2-y1, x2-x1))
                        else:
                            angle = np.pi/2
                        
                        # Must be within 20 degrees of vertical
                        if np.abs(angle - np.pi/2) < np.pi/9 and length > max_length:
                            max_length = length
                            best_line = (x1, y1, x2, y2)
                    
                    if best_line and max_length > 300:
                        x1, y1, x2, y2 = best_line
                        
                        # Draw detected line
                        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
                        
                        # Mark endpoints
                        top_y = min(y1, y2)
                        bottom_y = max(y1, y2)
                        center_x = (x1 + x2) // 2
                        
                        cv2.circle(image, (center_x, top_y), 8, (0, 0, 255), -1)
                        cv2.circle(image, (center_x, bottom_y), 8, (0, 255, 0), -1)
                        
                        cv2.putText(image, "TOP", (center_x + 15, top_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(image, "TIP", (center_x + 15, bottom_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Measure length
                        crutch_length_pixels = bottom_y - top_y
                        crutch_measurements.append(crutch_length_pixels)
                        frame_count += 1
                        
                        # Status
                        cv2.putText(image, f"Measuring: {frame_count}/30", 
                                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.8, (0, 255, 255), 2)
                        
                        if frame_count >= 30:
                            state = STATE_COMPLETE
                    else:
                        cv2.putText(image, "Crutch not detected - adjust position", 
                                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(image, "No lines detected - ensure good lighting", 
                               (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (0, 0, 255), 2)
            
            # ========================================
            # Display & Input
            # ========================================
            cv2.imshow("Crutch Scanner", image)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return None
            elif key == ord(' ') and state == STATE_READY_FOR_CRUTCH:
                state = STATE_SCANNING_CRUTCH
                frame_count = 0
                crutch_measurements = []
            
            if state == STATE_COMPLETE:
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # ========================================
    # Process Results
    # ========================================
    if len(crutch_measurements) < 10:
        print(f"\n❌ Could not measure crutch reliably")
        return None
    
    # Statistics
    avg_pixels = np.mean(crutch_measurements)
    std_pixels = np.std(crutch_measurements)
    
    # Convert to cm
    if pixels_per_cm is None:
        print(f"\n❌ No calibration available")
        return None
    
    length_cm = avg_pixels / pixels_per_cm
    stability_cm = std_pixels / pixels_per_cm
    
    print(f"\n✅ Crutch Scan Complete!")
    print(f"   Length: {length_cm:.1f} cm ({length_cm/2.54:.1f} inches)")
    print(f"   Stability: ±{stability_cm:.1f} cm")
    
    # Sanity check
    if length_cm < 90 or length_cm > 170:
        print(f"\n⚠️  WARNING: Measurement seems unusual")
        print(f"   Expected range: 90-170 cm")
        print(f"   Make sure crutch was at same distance as calibration")
        
        retry = input("\n   Try again? (y/n): ")
        if retry.lower() == 'y':
            return scan_crutch_simple(crutch_name, pixels_per_cm)
    
    return {
        'name': crutch_name,
        'length_cm': length_cm,
        'length_pixels': avg_pixels,
        'stability_cm': stability_cm,
        'pixels_per_cm': pixels_per_cm
    }

print("✓ Simple crutch scanner loaded")