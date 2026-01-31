"""
Calibration visualization
"""
import cv2
import numpy as np


def draw_calibration_overlay(image, phase, countdown=None, visibility_check=None):
    """
    Optimized calibration overlays - minimal copying, direct drawing
    """
    height, width = image.shape[:2]
    
    if phase == 'positioning':
        # Darken in-place (fast)
        image[:] = (image * 0.5).astype(np.uint8)
        
        # Header box
        header_y = 40
        header_h = 80
        cv2.rectangle(image, (width//2 - 300, header_y), 
                     (width//2 + 300, header_y + header_h), 
                     (50, 50, 50), -1)
        cv2.rectangle(image, (width//2 - 300, header_y), 
                     (width//2 + 300, header_y + header_h), 
                     (255, 255, 255), 2)
        
        cv2.putText(image, "Position yourself 5-7 feet away", 
                   (width//2 - 280, header_y + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Visibility checklist (compact)
        if visibility_check:
            y_start = 180
            checks = [
                ("Head", visibility_check.get('head', False)),
                ("Arms", visibility_check.get('arms', False)),
                ("Legs", visibility_check.get('legs', False)),
                ("Distance", visibility_check.get('distance', False))
            ]
            
            for i, (label, status) in enumerate(checks):
                color = (0, 255, 0) if status else (0, 100, 255)
                symbol = "✓" if status else "○"
                cv2.putText(image, f"{symbol} {label}", 
                           (width//2 - 100, y_start + i*35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return image
    
    elif phase == 'ready':
        # Light darken
        image[:] = (image * 0.7).astype(np.uint8)
        
        # Compact green box
        box_w, box_h = 350, 80
        cv2.rectangle(image, (width//2 - box_w//2, height//2 - box_h//2), 
                     (width//2 + box_w//2, height//2 + box_h//2), 
                     (30, 80, 30), -1)
        cv2.rectangle(image, (width//2 - box_w//2, height//2 - box_h//2), 
                     (width//2 + box_w//2, height//2 + box_h//2), 
                     (0, 255, 0), 3)
        
        cv2.putText(image, "Prepare to move...", 
                   (width//2 - 140, height//2 + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return image
    
    elif phase == 'countdown':
        # Medium darken
        image[:] = (image * 0.6).astype(np.uint8)
        
        # Giant countdown number
        text = str(countdown)
        font_scale = 6
        thickness = 12
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        # Single draw (no outline for performance)
        cv2.putText(image, text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        
        return image
    
    elif phase == 'calibrating':
        # Minimal overlay - just status bar
        bar_w = 450
        bar_h = 70
        cv2.rectangle(image, (width//2 - bar_w//2, 20), 
                     (width//2 + bar_w//2, 20 + bar_h), 
                     (40, 40, 40), -1)
        cv2.rectangle(image, (width//2 - bar_w//2, 20), 
                     (width//2 + bar_w//2, 20 + bar_h), 
                     (0, 255, 255), 2)
        
        cv2.putText(image, "CALIBRATING... Stand still!", 
                   (width//2 - 200, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return image
    
    return image