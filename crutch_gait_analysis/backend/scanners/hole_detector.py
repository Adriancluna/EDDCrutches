"""
Cell 5.6: Crutch Hole Detector & Counter
Detects adjustment holes and calculates device profile
"""

import cv2
import numpy as np

def scan_crutch_holes(crutch_name='right crutch', pixels_per_cm=None):
    """
    Advanced scanner: detects adjustment holes and measures spacing.
    
    Returns:
        dict with:
        - num_holes: number of adjustment holes detected
        - hole_spacing_cm: average distance between holes
        - overall_length_cm: total crutch length
        - min_setting_cm, max_setting_cm: range
    """
    
    print(f"\n{'='*70}")
    print(f"ADVANCED SCAN: {crutch_name.upper()} - HOLE DETECTION")
    print(f"{'='*70}")
    
    print(f"\n📸 Special Instructions for Hole Detection:")
    print(f"   1. Place crutch SIDEWAYS (holes should face camera)")
    print(f"   2. Ensure adjustment holes are CLEARLY visible")
    print(f"   3. Good lighting is CRITICAL")
    print(f"   4. White/light background helps")
    print(f"   5. Keep crutch steady")
    
    input("\n▶️  Press Enter when ready...")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n🎥 Capturing image...")
    print("   Adjusting exposure... (2 seconds)")
    
    # Let camera adjust
    for i in range(60):
        ret, frame = cap.read()
    
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera capture failed")
        cap.release()
        return None
    
    frame = cv2.flip(frame, 1)
    image = frame.copy()
    image_height, image_width = image.shape[:2]
    
    print("✓ Image captured")
    
    # ========================================
    # Step 1: Detect Crutch Shaft
    # ========================================
    print("\n🔍 Detecting crutch shaft...")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Find long lines (crutch shaft)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                           minLineLength=300, maxLineGap=30)
    
    crutch_shaft = None
    if lines is not None:
        # Find longest nearly-vertical line
        max_length = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            angle = np.abs(np.arctan2(y2-y1, x2-x1))
            if np.abs(angle - np.pi/2) < np.pi/6 and length > max_length:
                max_length = length
                crutch_shaft = (x1, y1, x2, y2)
    
    if crutch_shaft is None:
        print("❌ Could not detect crutch shaft")
        cap.release()
        cv2.destroyAllWindows()
        return None
    
    x1, y1, x2, y2 = crutch_shaft
    print(f"✓ Shaft detected (length: {max_length:.0f} pixels)")
    
    # ========================================
    # Step 2: Define Region of Interest (where holes are)
    # ========================================
    print("\n🎯 Analyzing hole region...")
    
    # Holes are typically in the lower 60% of the crutch
    shaft_top_y = min(y1, y2)
    shaft_bottom_y = max(y1, y2)
    shaft_center_x = (x1 + x2) // 2
    
    # ROI: around the shaft, lower portion
    roi_top = int(shaft_top_y + (shaft_bottom_y - shaft_top_y) * 0.3)
    roi_bottom = int(shaft_bottom_y)
    roi_left = max(0, shaft_center_x - 50)
    roi_right = min(image_width, shaft_center_x + 50)
    
    roi = gray[roi_top:roi_bottom, roi_left:roi_right]
    
    # ========================================
    # Step 3: Detect Circular Holes
    # ========================================
    print("🔍 Detecting holes...")
    
    # Apply adaptive threshold
    roi_thresh = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Detect circles (holes)
    circles = cv2.HoughCircles(
        roi_thresh, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=15,  # Minimum distance between hole centers
        param1=50, 
        param2=15,
        minRadius=3, 
        maxRadius=12
    )
    
    if circles is None:
        print("❌ No holes detected")
        print("   Try:")
        print("   - Better lighting")
        print("   - Ensure holes face camera")
        print("   - Light background")
        cap.release()
        cv2.destroyAllWindows()
        return None
    
    circles = np.uint16(np.around(circles))
    num_holes = circles.shape[1]
    
    print(f"✓ Detected {num_holes} holes")
    
    # ========================================
    # Step 4: Calculate Spacing
    # ========================================
    print("\n📏 Calculating hole spacing...")
    
    # Get hole centers (adjusted to full image coordinates)
    hole_centers_y = []
    for circle in circles[0, :]:
        cx, cy, r = circle
        # Adjust to full image coordinates
        full_y = cy + roi_top
        hole_centers_y.append(full_y)
    
    # Sort by Y position (top to bottom)
    hole_centers_y.sort()
    
    # Calculate spacings between consecutive holes
    spacings_pixels = []
    for i in range(len(hole_centers_y) - 1):
        spacing = hole_centers_y[i+1] - hole_centers_y[i]
        spacings_pixels.append(spacing)
    
    if len(spacings_pixels) == 0:
        print("❌ Not enough holes to calculate spacing")
        cap.release()
        cv2.destroyAllWindows()
        return None
    
    avg_spacing_pixels = np.mean(spacings_pixels)
    std_spacing_pixels = np.std(spacings_pixels)
    
    # Convert to cm
    if pixels_per_cm is None:
        print("⚠️  No calibration - using approximate conversion")
        pixels_per_cm = 10  # Rough estimate
    
    avg_spacing_cm = avg_spacing_pixels / pixels_per_cm
    
    print(f"✓ Average hole spacing: {avg_spacing_cm:.2f} cm ({avg_spacing_cm/2.54:.2f} inches)")
    print(f"  Consistency: ±{std_spacing_pixels/pixels_per_cm:.2f} cm")
    
    # ========================================
    # Step 5: Calculate Device Profile
    # ========================================
    print("\n🔧 Calculating device profile...")
    
    # Total length
    total_length_pixels = shaft_bottom_y - shaft_top_y
    total_length_cm = total_length_pixels / pixels_per_cm
    
    # Min/max settings (based on hole positions)
    min_setting_cm = hole_centers_y[0] / pixels_per_cm
    max_setting_cm = hole_centers_y[-1] / pixels_per_cm
    
    # ========================================
    # Visualization
    # ========================================
    print("\n📊 Generating visualization...")
    
    vis = image.copy()
    
    # Draw shaft
    cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Draw ROI
    cv2.rectangle(vis, (roi_left, roi_top), (roi_right, roi_bottom), 
                  (0, 255, 0), 2)
    
    # Draw detected holes
    for i, (circle, center_y) in enumerate(zip(circles[0, :], sorted(hole_centers_y))):
        cx, cy, r = circle
        # Adjust coordinates
        full_x = cx + roi_left
        full_y = cy + roi_top
        
        # Draw circle
        cv2.circle(vis, (full_x, full_y), r, (0, 0, 255), 2)
        
        # Label
        cv2.putText(vis, f"{i+1}", (full_x + 15, full_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # Info overlay
    info_y = 30
    cv2.putText(vis, f"Holes detected: {num_holes}", 
               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, f"Spacing: {avg_spacing_cm:.2f}cm", 
               (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, f"Range: {min_setting_cm:.0f}-{max_setting_cm:.0f}cm", 
               (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("Hole Detection Results", vis)
    print("\n✅ Review the detection (press any key to continue)")
    cv2.waitKey(0)
    
    cap.release()
    cv2.destroyAllWindows()
    
    # ========================================
    # Return Results
    # ========================================
    result = {
        'name': crutch_name,
        'num_holes': num_holes,
        'hole_spacing_cm': round(avg_spacing_cm, 2),
        'hole_spacing_inches': round(avg_spacing_cm / 2.54, 2),
        'overall_length_cm': round(total_length_cm, 1),
        'min_setting_cm': round(min_setting_cm, 1),
        'max_setting_cm': round(max_setting_cm, 1),
        'pixels_per_cm': pixels_per_cm
    }
    
    print(f"\n{'='*70}")
    print("SCAN COMPLETE")
    print(f"{'='*70}")
    print(f"\n📋 Device Profile:")
    for key, value in result.items():
        if key != 'pixels_per_cm':
            print(f"   {key}: {value}")
    
    return result

print("✓ Advanced hole detector loaded")