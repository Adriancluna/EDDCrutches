"""
Crutch Gait Analysis - Main Application
Run this to start the analysis: python main.py
"""
import cv2
import mediapipe as mp
import numpy as np
import signal
import sys
import os
import time

# Import our modules
from config import CrutchDeviceProfile
from core.geometry import (
    compute_distance, compute_angle, compute_angle_from_vertical,
    compute_angle_3d, compute_angle_from_vertical_3d,
    check_visibility, extract_point_3d
)
from core.gait_detector import GaitPhaseDetector
from core.temporal_tracker import TemporalTracker
from core.evaluators import (
    evaluate_elbow, evaluate_knee, evaluate_trunk_lean,
    evaluate_shoulder_asym, evaluate_step_length, evaluate_base_width,
    calculate_step_ratio, calculate_base_ratio
)
from core.one_euro_filter import LandmarkFilter, create_balanced_filter
from core.crutch_detector import CrutchDetector
from core.trial_manager import TrialManager
from data.session_collector import SessionCollector
from data.session_analyzer import SessionAnalyzer
from ui.calibration import draw_calibration_overlay
from recommender.live_optimizer import LiveFitOptimizer

# MediaPipe setup (robust import for different mediapipe layouts)
try:
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
except Exception:
    try:
        from mediapipe import solutions as mp_solutions
        mp_drawing = mp_solutions.drawing_utils
        mp_pose = mp_solutions.pose
    except Exception:
        mp_drawing = None
        mp_pose = None


def cleanup_and_exit(cap, session_collector=None, trial_manager=None):
    """Properly release resources and exit"""
    print("\n🛑 Shutting down gracefully...")
    cap.release()
    cv2.destroyAllWindows()
    print("✓ Camera released")
    print("✓ Windows closed")

    # Print trial summary with recommendations
    if trial_manager:
        # Collect feedback for the last trial if not already done
        if trial_manager.trial_history:
            last_trial = trial_manager.trial_history[-1]
            if last_trial.get('user_rating') is None:
                print("\nBefore we finish, let's get feedback on your last trial...")
                trial_manager.end_trial_with_feedback()

        trial_manager.print_session_summary()

    if session_collector:
        filepath = session_collector.save_session()

        try:
            analyzer = SessionAnalyzer(filepath)
            severity = analyzer.get_issue_severity_score()
            top_issues = analyzer.get_top_issues(top_n=3)

            if severity < 10:
                grade, emoji = "A (Excellent)", "🟢"
            elif severity < 25:
                grade, emoji = "B (Good)", "🟡"
            elif severity < 50:
                grade, emoji = "C (Needs Improvement)", "🟠"
            else:
                grade, emoji = "D (Poor Form)", "🔴"

            print(f"\n{emoji} BIOMECHANICS SUMMARY:")
            print(f"   Grade: {grade}")
            print(f"   Issue Severity: {severity:.1f}%")

            if top_issues:
                print(f"\n   Top Issues:")
                issue_labels = {
                    'elbow_r': 'Right Elbow', 'elbow_l': 'Left Elbow',
                    'trunk': 'Trunk Lean', 'knee_r': 'Right Knee',
                    'knee_l': 'Left Knee', 'step': 'Step Length',
                    'base': 'Base Width', 'shoulder': 'Shoulder Asymmetry'
                }
                for i, (issue, frames, pct) in enumerate(top_issues, 1):
                    label = issue_labels.get(issue, issue)
                    print(f"   {i}. {label} ({pct:.1f}%)")
            else:
                print(f"\n   ✅ No persistent issues detected!")

        except Exception as e:
            print(f"⚠️  Could not generate biomechanics summary: {e}")

    sys.exit(0)


def main():
    """Main application loop"""
    global mp_pose, mp_drawing
    # If MediaPipe solutions API is not available, run in simulation (no camera/pose) mode.
    simulate_mediapipe = False
    if mp_pose is None or mp_drawing is None:
        simulate_mediapipe = True
        print("⚠️  Mediapipe 'solutions' API not available. Running in simulation mode.")
        print("   Real pose detection will be disabled; no camera processing will occur.")

        # Lightweight compatibility shim so rest of the code can import/run without AttributeError
        class _FakePoseLandmark:
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

        class _FakeResults:
            def __init__(self):
                self.pose_landmarks = None

        class _FakePose:
            def __init__(self, *args, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def process(self, image):
                return _FakeResults()

        class _DummyDrawing:
            @staticmethod
            def draw_landmarks(*args, **kwargs):
                return None

        mp_pose = type('mp_pose', (), {'PoseLandmark': _FakePoseLandmark, 'Pose': _FakePose})
        mp_drawing = _DummyDrawing()
    
    # ========================================
    # User Setup
    # ========================================
    print("\n" + "="*70)
    print("CRUTCH GAIT ANALYSIS - SETUP")
    print("="*70)

    # Get user/session identifier
    print("\n=== Session Identification ===")
    print("Enter a name or identifier for this session.")
    print("Examples: 'john', 'patient_001', 'test_run', 'calibration_check'")
    user_id = input("User/Session ID: ").strip()

    if not user_id:
        user_id = "anonymous"

    # Sanitize for folder name (replace spaces/special chars)
    user_id = "".join(c if c.isalnum() or c in '-_' else '_' for c in user_id).lower()

    print(f"Session data will be saved to: sessions/{user_id}/")

    # Get user height
    print("\n=== Enter Your Height ===")
    height_unit = input("Use (f)eet or (c)m? ").strip().lower()
    
    if height_unit in ['f', 'feet', 'ft']:
        feet = float(input("Feet: "))
        inches = float(input("Inches: "))
        user_height_cm = (feet * 30.48) + (inches * 2.54)
    else:
        user_height_cm = float(input("Height (cm): "))
    
    expected_leg_length_cm = user_height_cm * 0.53

    print(f"\n✓ Your height: {user_height_cm:.1f} cm")
    print(f"✓ Expected leg length: {expected_leg_length_cm:.1f} cm")

    # Calculate body proportions for reference
    expected_wrist_height = user_height_cm * 0.485
    expected_axilla_height = user_height_cm * 0.815

    print(f"\nBody Proportions (estimated):")
    print(f"  Wrist height: {expected_wrist_height:.1f} cm")
    print(f"  Underarm height: {expected_axilla_height:.1f} cm")

    # Get user weight (optional, for AI recommendation)
    print("\n=== Enter Your Weight (Optional) ===")
    weight_input = input("Weight in kg (press Enter to skip): ").strip()
    if weight_input:
        try:
            user_weight_kg = float(weight_input)
            print(f"✓ Your weight: {user_weight_kg:.1f} kg")
        except ValueError:
            user_weight_kg = None
            print("  Skipped (will use default for AI recommendation)")
    else:
        user_weight_kg = None
        print("  Skipped (will use default for AI recommendation)")

    # ========================================
    # Crutch Specification (BEFORE AI recommendation)
    # ========================================
    print("\n" + "="*70)
    print("YOUR CRUTCH SPECIFICATIONS")
    print("="*70)
    print("\nTo give accurate recommendations, we need to know about YOUR crutches.")
    print("Different crutches have different numbers of adjustment positions.")

    # Get number of grip holes
    print("\n📍 HANDGRIP ADJUSTMENT:")
    print("   Count the holes on the forearm cuff section where the handle attaches.")
    print("   (Usually 6-12 holes, spaced about 1 inch apart)")
    while True:
        try:
            grip_holes_input = input("\n   Number of grip adjustment holes: ").strip()
            grip_num_holes = int(grip_holes_input)
            if 3 <= grip_num_holes <= 20:
                break
            else:
                print("   Please enter a number between 3 and 20")
        except ValueError:
            print("   Please enter a valid number")

    # Get grip height range or estimate
    print(f"\n   Do you know the grip height range of your crutch?")
    print("   (Check the manual or measure from floor to lowest/highest grip position)")
    grip_range_known = input("   Know the range? (y/n): ").strip().lower()

    if grip_range_known == 'y':
        try:
            grip_min = float(input("   Lowest grip height (cm): ").strip())
            grip_max = float(input("   Highest grip height (cm): ").strip())
        except ValueError:
            # Estimate based on user height
            grip_min = user_height_cm * 0.43  # ~43% of height
            grip_max = user_height_cm * 0.53  # ~53% of height
            print(f"   Using estimated range: {grip_min:.1f} - {grip_max:.1f} cm")
    else:
        # Estimate typical range based on user height
        grip_min = user_height_cm * 0.43
        grip_max = user_height_cm * 0.53
        print(f"   Using estimated range: {grip_min:.1f} - {grip_max:.1f} cm")

    # Get number of overall positions
    print("\n📍 OVERALL LENGTH ADJUSTMENT:")
    print("   Count the push-button positions on the lower tube section.")
    print("   (Usually 4-8 positions)")
    while True:
        try:
            overall_positions_input = input("\n   Number of overall length positions: ").strip()
            overall_num_positions = int(overall_positions_input)
            if 2 <= overall_num_positions <= 15:
                break
            else:
                print("   Please enter a number between 2 and 15")
        except ValueError:
            print("   Please enter a valid number")

    # Get overall length range or estimate
    print(f"\n   Do you know the overall length range of your crutch?")
    overall_range_known = input("   Know the range? (y/n): ").strip().lower()

    if overall_range_known == 'y':
        try:
            overall_min = float(input("   Shortest overall length (cm): ").strip())
            overall_max = float(input("   Longest overall length (cm): ").strip())
        except ValueError:
            # Estimate based on user height
            overall_min = user_height_cm * 0.65
            overall_max = user_height_cm * 0.80
            print(f"   Using estimated range: {overall_min:.1f} - {overall_max:.1f} cm")
    else:
        # Estimate typical range
        overall_min = user_height_cm * 0.65
        overall_max = user_height_cm * 0.80
        print(f"   Using estimated range: {overall_min:.1f} - {overall_max:.1f} cm")

    # Create custom device profile for this user's crutch
    user_crutch_profile = CrutchDeviceProfile(
        name="User's Crutch",
        grip_range=(grip_min, grip_max),
        grip_positions=grip_num_holes,
        overall_range=(overall_min, overall_max),
        overall_positions=overall_num_positions,
        model_id="user_custom"
    )

    print(f"\n✓ Crutch profile created:")
    print(f"   Grip: {grip_num_holes} positions ({grip_min:.1f} - {grip_max:.1f} cm)")
    print(f"   Overall: {overall_num_positions} positions ({overall_min:.1f} - {overall_max:.1f} cm)")
    print(f"   Grip step: ~{user_crutch_profile.grip_step_cm:.2f} cm ({user_crutch_profile.grip_step_cm/2.54:.2f}\")")
    print(f"   Overall step: ~{user_crutch_profile.overall_step_cm:.2f} cm ({user_crutch_profile.overall_step_cm/2.54:.2f}\")")

    # ========================================
    # AI Recommendation (using user's crutch profile)
    # ========================================
    print("\n" + "="*70)
    print("AI RECOMMENDATION")
    print("="*70)

    # Try to load trained model
    model_path = 'models/knn_recommender.pkl'
    use_ai = False
    ai_grip_cm = None
    ai_overall_cm = None
    ai_grip_idx = None
    ai_overall_idx = None

    if os.path.exists(model_path):
        try:
            from recommender.knn_recommender import KNNRecommender

            print(f"\n🤖 Loading AI recommender...")
            recommender = KNNRecommender()
            recommender.load(model_path)

            # Get recommendation using the USER'S crutch profile
            recommendation = recommender.predict(user_height_cm, user_weight_kg, device_profile=user_crutch_profile)

            print(f"\n🎯 AI RECOMMENDED INITIAL SETTINGS:")
            print(f"   Based on: {recommendation['user_height_cm']:.0f}cm, {recommendation['user_weight_kg']:.0f}kg")
            print(f"   Using YOUR crutch: {grip_num_holes} grip holes, {overall_num_positions} overall positions")
            print(f"\n   Handgrip: Hole #{recommendation['grip_setting_number']} of {grip_num_holes}")
            print(f"             ({recommendation['grip_height_cm']:.1f}cm / {recommendation['grip_height_cm']/2.54:.1f}\")")
            print(f"             Confidence: {recommendation['grip_confidence']*100:.0f}%")
            print(f"\n   Overall:  Position #{recommendation['overall_setting_number']} of {overall_num_positions}")
            print(f"             ({recommendation['overall_length_cm']:.1f}cm / {recommendation['overall_length_cm']/2.54:.1f}\")")
            print(f"             Confidence: {recommendation['overall_confidence']*100:.0f}%")

            # Ask user
            use_ai_input = input(f"\nUse AI recommendation? (y/n): ").strip().lower()

            if use_ai_input == 'y':
                ai_grip_cm = recommendation['grip_height_cm']
                ai_overall_cm = recommendation['overall_length_cm']
                ai_grip_idx = recommendation['grip_setting_idx']
                ai_overall_idx = recommendation['overall_setting_idx']
                use_ai = True
                print("✅ Using AI recommendation")
            else:
                print("✅ Will use manual input")

        except Exception as e:
            print(f"⚠️  Could not load AI recommender: {e}")
            print("   Falling back to manual input")

    else:
        print(f"\n💡 AI recommender not found at {model_path}")
        print("   To train: python scripts/train_recommender.py")
        print("   Using manual input for now")

    # ========================================
    # Crutch Configuration Method Selection
    # ========================================
    print("\n" + "="*70)
    print("CRUTCH CONFIGURATION")
    print("="*70)

    # Variables to store crutch config
    right_crutch_config = None
    left_crutch_config = None
    scanned_pixels_per_cm = None

    # If AI recommendation was accepted, use those values
    if use_ai and ai_grip_cm is not None and ai_overall_cm is not None:
        print(f"\n✅ Using AI-recommended settings:")
        print(f"   Grip height: {ai_grip_cm:.1f} cm")
        print(f"   Overall length: {ai_overall_cm:.1f} cm")

        right_crutch_config = {
            'side': 'right',
            'length_cm': ai_overall_cm,
            'handle_height_cm': ai_grip_cm,
            'hole_count': None,
            'hole_spacing_cm': None,
            'source': 'ai_recommendation'
        }
        left_crutch_config = right_crutch_config.copy()
        left_crutch_config['side'] = 'left'

        input_method = 'ai'  # Mark as AI-configured
    else:
        print("\nHow would you like to input your crutch settings?")
        print("  1. Manual entry (enter measurements directly)")
        print("  2. Camera scan (detect adjustment holes automatically)")

        input_method = input("\nSelect method (1 or 2): ").strip()

    if input_method == '2':
        # Camera scanning mode
        print("\n" + "-"*50)
        print("CRUTCH SCANNING MODE")
        print("-"*50)
        print("\nThis will use your camera to detect the adjustment holes")
        print("on your crutches and measure:")
        print("  - Number of adjustment positions")
        print("  - Distance between holes")
        print("  - Total adjustable range")
        print("\nYou'll show the SIDE of each crutch so the holes are visible.")

        try:
            from scanners.hole_scanner import scan_crutch_holes

            # Scan right crutch
            print("\nStarting with the RIGHT crutch...")
            input("Press Enter when ready to begin scanning...")

            right_crutch_config = scan_crutch_holes(
                side='right',
                pixels_per_cm=None,
                user_height_cm=user_height_cm
            )

            if right_crutch_config:
                scanned_pixels_per_cm = right_crutch_config.get('pixels_per_cm')

                print(f"\nRight crutch scanned successfully!")
                print(f"  Overall length: {right_crutch_config['length_cm']:.1f} cm")
                print(f"  Handle height: {right_crutch_config['handle_height_cm']:.1f} cm")
                print(f"  Adjustment holes: {right_crutch_config['hole_count']}")
                print(f"  Hole spacing: {right_crutch_config['hole_spacing_cm']:.2f} cm")

                # Ask about left crutch
                scan_left = input("\nScan left crutch too? (y/n): ").strip().lower()

                if scan_left == 'y':
                    left_crutch_config = scan_crutch_holes(
                        side='left',
                        pixels_per_cm=scanned_pixels_per_cm,
                        user_height_cm=user_height_cm
                    )

                    if not left_crutch_config:
                        print("Left crutch scan failed. Assuming same as right.")
                        left_crutch_config = right_crutch_config.copy()
                        left_crutch_config['side'] = 'left'
                else:
                    print("Assuming left crutch matches right crutch.")
                    left_crutch_config = right_crutch_config.copy()
                    left_crutch_config['side'] = 'left'
            else:
                print("\nScan failed. Falling back to manual input...")
                input_method = '1'

        except ImportError as e:
            print(f"\nScanner module not available: {e}")
            print("Falling back to manual input...")
            input_method = '1'
        except Exception as e:
            print(f"\nScanning error: {e}")
            print("Falling back to manual input...")
            input_method = '1'

    if input_method not in ['2', 'ai'] or right_crutch_config is None:
        # Manual input mode
        print("\n" + "-"*50)
        print("MANUAL ENTRY MODE")
        print("-"*50)
        print("\nHow do you want to enter your crutch settings?")
        print("  1. By hole/position number (recommended)")
        print("  2. By measurement in cm")

        entry_method = input("\nSelect (1 or 2): ").strip()

        if entry_method == '1':
            # Entry by hole number (using user's crutch profile)
            print("\n📋 Enter your CURRENT hole positions:")
            print(f"   (Using your crutch profile)")
            print(f"   Overall: {user_crutch_profile.overall_num_settings} positions available")
            print(f"   Grip: {user_crutch_profile.grip_num_settings} positions available")

            # Get overall position
            while True:
                try:
                    overall_input = input(f"\nOverall length position (1-{user_crutch_profile.overall_num_settings}): ").strip()
                    if not overall_input:
                        overall_setting_idx = user_crutch_profile.overall_num_settings // 2  # Default to middle
                        print(f"   Using default: position {overall_setting_idx + 1}")
                    else:
                        overall_setting_num = int(overall_input)
                        if 1 <= overall_setting_num <= user_crutch_profile.overall_num_settings:
                            overall_setting_idx = overall_setting_num - 1
                        else:
                            print(f"   Please enter 1-{user_crutch_profile.overall_num_settings}")
                            continue
                    break
                except ValueError:
                    print("   Please enter a number")

            # Get grip position
            while True:
                try:
                    grip_input = input(f"Grip/handle position (1-{user_crutch_profile.grip_num_settings}): ").strip()
                    if not grip_input:
                        grip_setting_idx = user_crutch_profile.grip_num_settings // 2  # Default to middle
                        print(f"   Using default: position {grip_setting_idx + 1}")
                    else:
                        grip_setting_num = int(grip_input)
                        if 1 <= grip_setting_num <= user_crutch_profile.grip_num_settings:
                            grip_setting_idx = grip_setting_num - 1
                        else:
                            print(f"   Please enter 1-{user_crutch_profile.grip_num_settings}")
                            continue
                    break
                except ValueError:
                    print("   Please enter a number")

            # Convert to cm using user's crutch profile
            current_overall = user_crutch_profile.setting_to_cm(overall_setting_idx, 'overall')
            current_grip = user_crutch_profile.setting_to_cm(grip_setting_idx, 'grip')

            print(f"\n✓ Overall: Position {overall_setting_idx + 1} = {current_overall:.1f} cm")
            print(f"✓ Grip: Position {grip_setting_idx + 1} = {current_grip:.1f} cm")

        else:
            # Entry by cm measurement
            print("\nEnter your CURRENT crutch measurements:")
            print("(We'll track changes from this starting point)")

            current_overall = float(input("Overall crutch height in cm (floor to top): ") or user_height_cm - 40)
            current_grip = float(input("Handle height in cm (floor to handle): ") or user_height_cm * 0.485)

        right_crutch_config = {
            'side': 'right',
            'length_cm': current_overall,
            'handle_height_cm': current_grip,
            'hole_count': None,
            'hole_spacing_cm': None
        }
        left_crutch_config = right_crutch_config.copy()
        left_crutch_config['side'] = 'left'

    # Extract values for trial manager
    current_overall = right_crutch_config['length_cm']
    current_grip = right_crutch_config['handle_height_cm']

    # ========================================
    # Configuration Summary
    # ========================================
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)

    print(f"\nCurrent Settings:")
    print(f"  Right crutch length: {right_crutch_config['length_cm']:.1f} cm")
    print(f"  Left crutch length:  {left_crutch_config['length_cm']:.1f} cm")
    print(f"  Handle height: {right_crutch_config['handle_height_cm']:.1f} cm")

    if right_crutch_config.get('hole_count'):
        print(f"\nScanned crutch info:")
        print(f"  Adjustment holes: {right_crutch_config['hole_count']}")
        print(f"  Hole spacing: {right_crutch_config['hole_spacing_cm']:.2f} cm ({right_crutch_config['hole_spacing_cm']/2.54:.2f} in)")
        print(f"  Adjustable range: {right_crutch_config.get('adjustable_range_cm', 0):.1f} cm")

    # Calculate ideal settings for comparison
    ideal_overall = expected_axilla_height - 5  # 2" clearance from underarm
    ideal_handle = expected_wrist_height

    print(f"\nIdeal Settings (starting point):")
    print(f"  Overall length: {ideal_overall:.1f} cm ({ideal_overall/2.54:.1f} inches)")
    print(f"  Handle height:  {ideal_handle:.1f} cm ({ideal_handle/2.54:.1f} inches)")

    # Calculate initial recommendations
    overall_diff = current_overall - ideal_overall

    if abs(overall_diff) > 2.54:  # More than 1 inch off
        if overall_diff > 0:
            print(f"\nInitial Recommendation:")
            print(f"  Your crutches may be {abs(overall_diff):.1f} cm too TALL")
            print(f"  Consider lowering by ~{abs(overall_diff)/2.54:.1f} inches")
            if right_crutch_config.get('hole_spacing_cm'):
                holes_to_adjust = int(abs(overall_diff) / right_crutch_config['hole_spacing_cm'])
                if holes_to_adjust > 0:
                    print(f"  (approximately {holes_to_adjust} hole positions)")
        else:
            print(f"\nInitial Recommendation:")
            print(f"  Your crutches may be {abs(overall_diff):.1f} cm too SHORT")
            print(f"  Consider raising by ~{abs(overall_diff)/2.54:.1f} inches")
            if right_crutch_config.get('hole_spacing_cm'):
                holes_to_adjust = int(abs(overall_diff) / right_crutch_config['hole_spacing_cm'])
                if holes_to_adjust > 0:
                    print(f"  (approximately {holes_to_adjust} hole positions)")
    else:
        print(f"\nYour crutch height looks good!")

    # Store crutch configuration for later use
    crutch_config = {
        'right': right_crutch_config,
        'left': left_crutch_config,
        'ideal_overall': ideal_overall,
        'ideal_handle': ideal_handle
    }

    # Create device profile from scanned data or use user's custom profile
    if right_crutch_config.get('hole_count'):
        # Use scanned crutch specifications (overrides user input)
        device_profile = CrutchDeviceProfile.from_scan(
            right_crutch_config,
            current_length_cm=current_overall
        )
        print(f"\nUsing scanned crutch profile:")
        print(f"  {device_profile.overall_num_settings} overall positions (step: {device_profile.overall_step_cm:.2f} cm)")
        print(f"  {device_profile.grip_num_settings} grip positions (step: {device_profile.grip_step_cm:.2f} cm)")
    else:
        # Use the user's custom crutch profile (entered earlier)
        device_profile = user_crutch_profile
        print(f"\nUsing your crutch profile:")
        print(f"  {device_profile.grip_num_settings} grip positions (step: {device_profile.grip_step_cm:.2f} cm)")
        print(f"  {device_profile.overall_num_settings} overall positions (step: {device_profile.overall_step_cm:.2f} cm)")

    print(f"\nSetup complete! Ready for calibration.")

    trial_manager = TrialManager(device_profile, current_overall, current_grip)

    # Initialize live optimizer for trial analysis
    live_optimizer = LiveFitOptimizer()
    print("✓ Live optimizer initialized")

    # Trial frame tracking for analysis
    current_trial_frames = []
    trial_scores = []  # Track score history across trials

    print("\n" + "="*70)
    print("Ready to start! Press 'q' when done walking.")
    print("Controls: []: grip  -/=: overall  N: analyze+new  B: best  C: crutch  Q: quit")
    print("="*70)
    
    # ========================================
    # Initialize Components
    # ========================================
    
    # Performance optimization: reduce resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Signal handler
    def signal_handler(sig, frame):
        cleanup_and_exit(cap, session_collector, trial_manager)

    signal.signal(signal.SIGINT, signal_handler)
    
    # Create named window
    cv2.namedWindow("Mediapipe Feed", cv2.WINDOW_NORMAL)
    
    # Initialize trackers
    tracker = TemporalTracker(window_size=30)
    phase_detector = GaitPhaseDetector(history_size=15)
    session_collector = None

    # Initialize One Euro Filter for landmark smoothing (new!)
    # This provides adaptive smoothing - more responsive when moving, smoother when still
    landmark_filter = create_balanced_filter(rate=30.0)
    print("One Euro Filter initialized for landmark smoothing")

    # Initialize crutch detector
    crutch_detector = CrutchDetector(history_size=10)
    crutch_detection_enabled = True  # Can be toggled with 'c' key
    print("Crutch detector initialized")
    
    # Calibration state
    calibration_complete = False
    calibration_phase = 'positioning'
    calibration_frames = []
    pixels_per_cm = None
    countdown_start_time = None
    countdown_value = 3
    positioning_frames_good = 0

    # Visibility threshold for landmark confidence
    # Increased from 0.5 to 0.7 to reduce false positives from MediaPipe estimation
    VISIBILITY_THRESHOLD = 0.7

    # Hip width calibration (for normalized base width ratio)
    calibrated_hip_width_cm = None
    hip_width_samples = []
    
    # Pre-define landmark indices for faster access
    NOSE = mp_pose.PoseLandmark.NOSE.value
    LEFT_WRIST = mp_pose.PoseLandmark.LEFT_WRIST.value
    RIGHT_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST.value
    LEFT_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE.value
    RIGHT_ANKLE = mp_pose.PoseLandmark.RIGHT_ANKLE.value
    LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
    RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value
    LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    LEFT_ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW.value
    RIGHT_ELBOW = mp_pose.PoseLandmark.RIGHT_ELBOW.value
    LEFT_KNEE = mp_pose.PoseLandmark.LEFT_KNEE.value
    RIGHT_KNEE = mp_pose.PoseLandmark.RIGHT_KNEE.value
    LEFT_FOOT_INDEX = mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value
    RIGHT_FOOT_INDEX = mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value
    
    # ========================================
    # Main Loop
    # ========================================
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert BGR → RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            
            # Convert back RGB → BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Get image dimensions once
            image_height, image_width = image.shape[:2]
            
            # ========================================
            # CALIBRATION FLOW
            # ========================================
            
            if not calibration_complete:
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Fast visibility checks
                    head_visible = landmarks[NOSE].visibility > 0.5
                    arms_visible = (landmarks[LEFT_WRIST].visibility > 0.5 and 
                                   landmarks[RIGHT_WRIST].visibility > 0.5)
                    legs_visible = (landmarks[LEFT_ANKLE].visibility > 0.5 and 
                                   landmarks[RIGHT_ANKLE].visibility > 0.5)
                    
                    # Distance check
                    hip_y = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) * 0.5
                    ankle_y = (landmarks[LEFT_ANKLE].y + landmarks[RIGHT_ANKLE].y) * 0.5
                    body_height_pixels = abs((ankle_y - hip_y) * image_height)
                    good_distance = 200 < body_height_pixels < 500
                    
                    all_checks_pass = head_visible and arms_visible and legs_visible and good_distance
                    
                    visibility_status = {
                        'head': head_visible,
                        'arms': arms_visible,
                        'legs': legs_visible,
                        'distance': good_distance
                    }
                    
                    # PHASE 1: POSITIONING
                    if calibration_phase == 'positioning':
                        image = draw_calibration_overlay(image, 'positioning', 
                                                         visibility_check=visibility_status)
                        
                        if all_checks_pass:
                            positioning_frames_good += 1
                            if positioning_frames_good >= 30:
                                calibration_phase = 'ready'
                                countdown_start_time = cv2.getTickCount()
                        else:
                            positioning_frames_good = 0
                    
                    # PHASE 2: READY
                    elif calibration_phase == 'ready':
                        image = draw_calibration_overlay(image, 'ready')
                        
                        elapsed = (cv2.getTickCount() - countdown_start_time) / cv2.getTickFrequency()
                        if elapsed >= 1.5:
                            calibration_phase = 'countdown'
                            countdown_start_time = cv2.getTickCount()
                            countdown_value = 3
                    
                    # PHASE 3: COUNTDOWN
                    elif calibration_phase == 'countdown':
                        elapsed = (cv2.getTickCount() - countdown_start_time) / cv2.getTickFrequency()
                        countdown_value = 3 - int(elapsed)
                        
                        if countdown_value > 0:
                            image = draw_calibration_overlay(image, 'countdown', 
                                                             countdown=countdown_value)
                        else:
                            calibration_phase = 'calibrating'
                    
                    # PHASE 4: CALIBRATING
                    elif calibration_phase == 'calibrating':
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, 
                                                 mp_pose.POSE_CONNECTIONS)
                        
                        # Only extract 4 landmarks needed for calibration
                        lh = landmarks[LEFT_HIP]
                        rh = landmarks[RIGHT_HIP]
                        la = landmarks[LEFT_ANKLE]
                        ra = landmarks[RIGHT_ANKLE]
                        
                        lx_hip, ly_hip = lh.x * image_width, lh.y * image_height
                        rx_hip, ry_hip = rh.x * image_width, rh.y * image_height
                        lx_ankle, ly_ankle = la.x * image_width, la.y * image_height
                        rx_ankle, ry_ankle = ra.x * image_width, ra.y * image_height
                        
                        # Measure leg length
                        left_leg_pixels = compute_distance((lx_hip, ly_hip), (lx_ankle, ly_ankle))
                        right_leg_pixels = compute_distance((rx_hip, ry_hip), (rx_ankle, ry_ankle))
                        avg_leg_pixels = (left_leg_pixels + right_leg_pixels) * 0.5
                        
                        calibration_frames.append(avg_leg_pixels)
                        
                        # Show calibrating overlay
                        image = draw_calibration_overlay(image, 'calibrating')
                        cv2.putText(image, f"Frames: {len(calibration_frames)}/30", 
                            (image_width//2 - 80, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 255, 255), 2)
                        
                        if len(calibration_frames) >= 30:
                            avg_calibration_pixels = np.mean(calibration_frames)
                            pixels_per_cm = avg_calibration_pixels / expected_leg_length_cm
                            calibration_complete = True
                            
                            # Initialize session collector
                            session_collector = SessionCollector(user_height_cm, expected_leg_length_cm, user_id=user_id)
                            print(f"✓ Session data collection started (user: {user_id})")
                            
                            calibration_phase = 'complete'
                            print(f"\n✓ Calibration complete!")
                            print(f"  Conversion factor: {pixels_per_cm:.2f} pixels/cm")
                
                else:
                    # No pose detected
                    if calibration_phase == 'positioning':
                        image = draw_calibration_overlay(image, 'positioning', 
                                                         visibility_check={
                                                             'head': False,
                                                             'arms': False,
                                                             'legs': False,
                                                             'distance': False
                                                         })
                
                # Show and continue
                cv2.imshow("Mediapipe Feed", image)
                
                # Check if window was closed
                if cv2.getWindowProperty("Mediapipe Feed", cv2.WND_PROP_VISIBLE) < 1:
                    cleanup_and_exit(cap, session_collector, trial_manager)

                # Check for exit keys
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q') or key == 27:
                    cleanup_and_exit(cap, session_collector, trial_manager)

                continue
            
            # ========================================
            # MAIN ANALYSIS (After calibration)
            # ========================================

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks,
                                         mp_pose.POSE_CONNECTIONS)

                # Get both 2D and 3D landmarks
                landmarks = results.pose_landmarks.landmark
                world_landmarks = results.pose_world_landmarks.landmark if results.pose_world_landmarks else None

                # Get current timestamp for filtering
                current_time = time.time()

                # ========================================
                # Apply One Euro Filter to 2D landmarks
                # ========================================
                # Filter key landmarks for smoother position tracking
                key_indices = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
                              LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
                              LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
                              LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX]

                filtered_positions = {}
                for idx in key_indices:
                    lm = landmarks[idx]
                    # Only filter if visible
                    if lm.visibility >= VISIBILITY_THRESHOLD:
                        fx, fy = landmark_filter.filter_landmark(
                            idx, lm.x * image_width, lm.y * image_height,
                            timestamp=current_time
                        )
                        filtered_positions[idx] = (fx, fy)
                    else:
                        # Use raw position for non-visible landmarks
                        filtered_positions[idx] = (lm.x * image_width, lm.y * image_height)

                # Extract filtered pixel coordinates
                lx_shoulder, ly_shoulder = filtered_positions[LEFT_SHOULDER]
                rx_shoulder, ry_shoulder = filtered_positions[RIGHT_SHOULDER]

                lx_elbow, ly_elbow = filtered_positions[LEFT_ELBOW]
                rx_elbow, ry_elbow = filtered_positions[RIGHT_ELBOW]

                lx_wrist, ly_wrist = filtered_positions[LEFT_WRIST]
                rx_wrist, ry_wrist = filtered_positions[RIGHT_WRIST]

                lx_hip, ly_hip = filtered_positions[LEFT_HIP]
                rx_hip, ry_hip = filtered_positions[RIGHT_HIP]

                lx_knee, ly_knee = filtered_positions[LEFT_KNEE]
                rx_knee, ry_knee = filtered_positions[RIGHT_KNEE]

                lx_ankle, ly_ankle = filtered_positions[LEFT_ANKLE]
                rx_ankle, ry_ankle = filtered_positions[RIGHT_ANKLE]

                lx_foot, ly_foot = filtered_positions[LEFT_FOOT_INDEX]
                rx_foot, ry_foot = filtered_positions[RIGHT_FOOT_INDEX]

                # ========================================
                # Check visibility for each body part
                # ========================================
                # Basic visibility check from MediaPipe confidence scores
                right_arm_visible = check_visibility(landmarks,
                    [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST], VISIBILITY_THRESHOLD)
                left_arm_visible = check_visibility(landmarks,
                    [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST], VISIBILITY_THRESHOLD)
                right_leg_visible = check_visibility(landmarks,
                    [RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE], VISIBILITY_THRESHOLD)
                left_leg_visible = check_visibility(landmarks,
                    [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE], VISIBILITY_THRESHOLD)
                trunk_visible = check_visibility(landmarks,
                    [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP], VISIBILITY_THRESHOLD)
                feet_visible = check_visibility(landmarks,
                    [LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX, LEFT_ANKLE, RIGHT_ANKLE], VISIBILITY_THRESHOLD)

                # Additional sanity checks - landmarks must be within frame
                # (MediaPipe can estimate positions outside frame with moderate confidence)
                def in_frame(lm, margin=0.05):
                    """Check if landmark is within frame bounds with margin"""
                    return (margin < lm.x < 1.0 - margin and
                            margin < lm.y < 1.0 - margin)

                # Check if legs are actually in frame (not just estimated)
                right_leg_in_frame = (in_frame(landmarks[RIGHT_HIP]) and
                                      in_frame(landmarks[RIGHT_KNEE]) and
                                      in_frame(landmarks[RIGHT_ANKLE]))
                left_leg_in_frame = (in_frame(landmarks[LEFT_HIP]) and
                                     in_frame(landmarks[LEFT_KNEE]) and
                                     in_frame(landmarks[LEFT_ANKLE]))
                feet_in_frame = (in_frame(landmarks[LEFT_ANKLE]) and
                                 in_frame(landmarks[RIGHT_ANKLE]))

                # Anatomical sanity check: knee should be below hip, ankle below knee
                right_leg_sane = (landmarks[RIGHT_KNEE].y > landmarks[RIGHT_HIP].y and
                                  landmarks[RIGHT_ANKLE].y > landmarks[RIGHT_KNEE].y)
                left_leg_sane = (landmarks[LEFT_KNEE].y > landmarks[LEFT_HIP].y and
                                 landmarks[LEFT_ANKLE].y > landmarks[LEFT_KNEE].y)

                # Combine all checks
                right_leg_visible = right_leg_visible and right_leg_in_frame and right_leg_sane
                left_leg_visible = left_leg_visible and left_leg_in_frame and left_leg_sane
                feet_visible = feet_visible and feet_in_frame and (right_leg_sane or left_leg_sane)
                
                # ========================================
                # Compute distances (using 2D + calibration)
                # ========================================

                # Leg lengths
                left_leg_length_px = compute_distance((lx_hip, ly_hip), (lx_ankle, ly_ankle))
                right_leg_length_px = compute_distance((rx_hip, ry_hip), (rx_ankle, ry_ankle))

                # Convert to cm
                inv_pixels_per_cm = 1.0 / pixels_per_cm
                left_leg_length = left_leg_length_px * inv_pixels_per_cm
                right_leg_length = right_leg_length_px * inv_pixels_per_cm
                avg_leg_length = (left_leg_length + right_leg_length) * 0.5

                # Hip width (for normalized base width ratio)
                hip_width_px = compute_distance((lx_hip, ly_hip), (rx_hip, ry_hip))
                hip_width_cm = hip_width_px * inv_pixels_per_cm

                # Calibrate hip width over first few frames
                if calibrated_hip_width_cm is None and len(hip_width_samples) < 30:
                    hip_width_samples.append(hip_width_cm)
                    if len(hip_width_samples) >= 30:
                        calibrated_hip_width_cm = np.mean(hip_width_samples)
                        print(f"Hip width calibrated: {calibrated_hip_width_cm:.1f} cm")

                # Step length (only if feet visible)
                if feet_visible:
                    step_length_px = abs(rx_ankle - lx_ankle)
                    step_length = step_length_px * inv_pixels_per_cm
                    # Calculate normalized ratio
                    step_ratio = calculate_step_ratio(step_length, expected_leg_length_cm)
                else:
                    step_length = None
                    step_ratio = None

                # Base of support (only if feet visible)
                if feet_visible:
                    base_of_support_px = compute_distance((lx_foot, ly_foot), (rx_foot, ry_foot))
                    base_of_support = base_of_support_px * inv_pixels_per_cm
                    # Calculate normalized ratio (use calibrated or current hip width)
                    effective_hip_width = calibrated_hip_width_cm if calibrated_hip_width_cm else hip_width_cm
                    base_ratio = calculate_base_ratio(base_of_support, effective_hip_width)
                else:
                    base_of_support = None
                    base_ratio = None

                # ========================================
                # Compute angles (using 3D world landmarks when available)
                # ========================================

                # RIGHT ELBOW - use 3D if visible
                if right_arm_visible and world_landmarks:
                    right_elbow_angle = compute_angle_3d(
                        extract_point_3d(world_landmarks[RIGHT_SHOULDER]),
                        extract_point_3d(world_landmarks[RIGHT_ELBOW]),
                        extract_point_3d(world_landmarks[RIGHT_WRIST])
                    )
                elif right_arm_visible:
                    # Fallback to 2D
                    right_elbow_angle = compute_angle(
                        (rx_shoulder, ry_shoulder),
                        (rx_elbow, ry_elbow),
                        (rx_wrist, ry_wrist)
                    )
                else:
                    right_elbow_angle = None

                # LEFT ELBOW - use 3D if visible
                if left_arm_visible and world_landmarks:
                    left_elbow_angle = compute_angle_3d(
                        extract_point_3d(world_landmarks[LEFT_SHOULDER]),
                        extract_point_3d(world_landmarks[LEFT_ELBOW]),
                        extract_point_3d(world_landmarks[LEFT_WRIST])
                    )
                elif left_arm_visible:
                    left_elbow_angle = compute_angle(
                        (lx_shoulder, ly_shoulder),
                        (lx_elbow, ly_elbow),
                        (lx_wrist, ly_wrist)
                    )
                else:
                    left_elbow_angle = None

                # RIGHT KNEE - use 3D if visible
                if right_leg_visible and world_landmarks:
                    right_knee_angle = compute_angle_3d(
                        extract_point_3d(world_landmarks[RIGHT_HIP]),
                        extract_point_3d(world_landmarks[RIGHT_KNEE]),
                        extract_point_3d(world_landmarks[RIGHT_ANKLE])
                    )
                elif right_leg_visible:
                    right_knee_angle = compute_angle(
                        (rx_hip, ry_hip),
                        (rx_knee, ry_knee),
                        (rx_ankle, ry_ankle)
                    )
                else:
                    right_knee_angle = None

                # LEFT KNEE - use 3D if visible
                if left_leg_visible and world_landmarks:
                    left_knee_angle = compute_angle_3d(
                        extract_point_3d(world_landmarks[LEFT_HIP]),
                        extract_point_3d(world_landmarks[LEFT_KNEE]),
                        extract_point_3d(world_landmarks[LEFT_ANKLE])
                    )
                elif left_leg_visible:
                    left_knee_angle = compute_angle(
                        (lx_hip, ly_hip),
                        (lx_knee, ly_knee),
                        (lx_ankle, ly_ankle)
                    )
                else:
                    left_knee_angle = None

                # TRUNK LEAN - use 3D if visible
                if trunk_visible and world_landmarks:
                    # 3D midpoints
                    mid_shoulder_3d = (
                        (world_landmarks[LEFT_SHOULDER].x + world_landmarks[RIGHT_SHOULDER].x) / 2,
                        (world_landmarks[LEFT_SHOULDER].y + world_landmarks[RIGHT_SHOULDER].y) / 2,
                        (world_landmarks[LEFT_SHOULDER].z + world_landmarks[RIGHT_SHOULDER].z) / 2
                    )
                    mid_hip_3d = (
                        (world_landmarks[LEFT_HIP].x + world_landmarks[RIGHT_HIP].x) / 2,
                        (world_landmarks[LEFT_HIP].y + world_landmarks[RIGHT_HIP].y) / 2,
                        (world_landmarks[LEFT_HIP].z + world_landmarks[RIGHT_HIP].z) / 2
                    )
                    trunk_lean_angle = compute_angle_from_vertical_3d(mid_shoulder_3d, mid_hip_3d)
                elif trunk_visible:
                    mid_hip_x = (lx_hip + rx_hip) * 0.5
                    mid_hip_y = (ly_hip + ry_hip) * 0.5
                    mid_shoulder_x = (lx_shoulder + rx_shoulder) * 0.5
                    mid_shoulder_y = (ly_shoulder + ry_shoulder) * 0.5
                    trunk_lean_angle = compute_angle_from_vertical(
                        (mid_shoulder_x, mid_shoulder_y),
                        (mid_hip_x, mid_hip_y)
                    )
                else:
                    trunk_lean_angle = None

                # SHOULDER ASYMMETRY (2D is fine for this)
                if trunk_visible:
                    shoulder_height_diff = abs(ly_shoulder - ry_shoulder)
                    shoulder_elevation_percent = (shoulder_height_diff / avg_leg_length) * 100
                else:
                    shoulder_elevation_percent = None
                
                # ========================================
                # UPDATE GAIT PHASE DETECTOR 
                # ========================================
                
                phase_detector.update({
                    'rx_ankle': rx_ankle,
                    'ry_ankle': ry_ankle,
                    'lx_ankle': lx_ankle,
                    'ly_ankle': ly_ankle,
                    'rx_foot': rx_foot,
                    'ry_foot': ry_foot,
                    'lx_foot': lx_foot,
                    'ly_foot': ly_foot,
                    'rx_hip': rx_hip,
                    'ry_hip': ry_hip,
                    'lx_hip': lx_hip,
                    'ly_hip': ly_hip
                })
                
                current_phase = phase_detector.get_phase()
                phase_confidence = phase_detector.get_phase_confidence()

                # ========================================
                # CRUTCH DETECTION
                # ========================================
                crutch_detections = None
                crutch_evaluation = None

                # Only run crutch detection if:
                # 1. Feature is enabled
                # 2. Feet are visible (user is standing, not sitting)
                # 3. We have a valid floor estimate
                user_is_standing = feet_visible and current_phase != "STANDING"

                if crutch_detection_enabled and feet_visible:
                    # Estimate floor level from ankle positions
                    floor_y = int(max(ly_ankle, ry_ankle)) + 20

                    # Detect crutches
                    crutch_detections = crutch_detector.detect(
                        frame=image,
                        landmarks=landmarks,
                        image_width=image_width,
                        image_height=image_height,
                        floor_y=floor_y,
                        pixels_per_cm=pixels_per_cm
                    )

                    # Evaluate crutch placement (only if detection is confident)
                    if crutch_detections['confidence'] > 0.5:  # Increased from 0.3
                        crutch_evaluation = crutch_detector.evaluate_placement(
                            crutch_detections,
                            hip_width_px=hip_width_px,
                            pixels_per_cm=pixels_per_cm
                        )
                elif crutch_detection_enabled and not feet_visible:
                    # Reset detector when feet not visible to avoid stale data
                    crutch_detector.reset()

                # ========================================
                # EVALUATE MEASUREMENTS
                # ========================================

                # Use normalized ratios for step/base (body-size independent)
                evaluations = {
                    'elbow_r': evaluate_elbow(right_elbow_angle, 'right', current_phase),
                    'elbow_l': evaluate_elbow(left_elbow_angle, 'left', current_phase),
                    'knee_r': evaluate_knee(right_knee_angle, 'right'),
                    'knee_l': evaluate_knee(left_knee_angle, 'left'),
                    'trunk': evaluate_trunk_lean(trunk_lean_angle, current_phase),
                    'shoulder': evaluate_shoulder_asym(shoulder_elevation_percent),
                    'step': evaluate_step_length(length_cm=step_length, ratio=step_ratio, use_ratio=True),
                    'base': evaluate_base_width(width_cm=base_of_support, ratio=base_ratio, use_ratio=True)
                }
                
                # Update temporal tracker (only include valid measurements)
                measurements_to_track = {}
                if right_elbow_angle is not None:
                    measurements_to_track['elbow_r'] = right_elbow_angle
                if left_elbow_angle is not None:
                    measurements_to_track['elbow_l'] = left_elbow_angle
                if trunk_lean_angle is not None:
                    measurements_to_track['trunk_lean'] = trunk_lean_angle
                if right_knee_angle is not None:
                    measurements_to_track['knee_r'] = right_knee_angle
                if left_knee_angle is not None:
                    measurements_to_track['knee_l'] = left_knee_angle
                if step_length is not None:
                    measurements_to_track['step_length'] = step_length
                if base_of_support is not None:
                    measurements_to_track['base_width'] = base_of_support
                if shoulder_elevation_percent is not None:
                    measurements_to_track['shoulder_asym'] = shoulder_elevation_percent

                tracker.update(
                    measurements=measurements_to_track,
                    evaluations=evaluations
                )
                
                # ========================================
                # GENERATE SMART FEEDBACK
                # ========================================
                
                feedback_messages = []
                
                # Only report PERSISTENT issues
                persistent_issues = {
                    'elbow_r': tracker.is_issue_persistent('elbow_r', min_persistence=0.7),
                    'elbow_l': tracker.is_issue_persistent('elbow_l', min_persistence=0.7),
                    'trunk': tracker.is_issue_persistent('trunk', min_persistence=0.6),
                    'knee_r': tracker.is_issue_persistent('knee_r', min_persistence=0.7),
                    'knee_l': tracker.is_issue_persistent('knee_l', min_persistence=0.7),
                    'step': tracker.is_issue_persistent('step', min_persistence=0.5),
                    'base': tracker.is_issue_persistent('base', min_persistence=0.5),
                    'shoulder': tracker.is_issue_persistent('shoulder', min_persistence=0.6)
                }
                
                # Collect messages only for persistent issues
                for key, is_persistent in persistent_issues.items():
                    if is_persistent and evaluations[key]['message'] is not None:
                        feedback_messages.append(evaluations[key]['message'])
                
                # Sort by severity
                severity_order = {'critical': 0, 'warning': 1, 'caution': 2, 'good': 3}
                feedback_messages.sort(
                    key=lambda msg: min(
                        severity_order.get(evaluations[k]['status'], 3) 
                        for k in evaluations 
                        if evaluations[k].get('message') == msg
                    )
                )
                
                # ========================================
                # SESSION DATA COLLECTION
                # ========================================
                
                # Collect smoothed values
                smoothed_values = {
                    'elbow_r': tracker.get_smoothed('elbow_r'),
                    'elbow_l': tracker.get_smoothed('elbow_l'),
                    'trunk_lean': tracker.get_smoothed('trunk_lean'),
                    'knee_r': tracker.get_smoothed('knee_r'),
                    'knee_l': tracker.get_smoothed('knee_l'),
                    'step_length': tracker.get_smoothed('step_length'),
                    'base_width': tracker.get_smoothed('base_width'),
                    'shoulder_asym': tracker.get_smoothed('shoulder_asym')
                }
                
                # Get list of persistent issue names
                persistent_issue_names = [k for k, v in persistent_issues.items() if v]

                # Update trial manager with biomechanics data (handle None values)
                valid_elbows = [e for e in [right_elbow_angle, left_elbow_angle] if e is not None]
                avg_elbow = sum(valid_elbows) / len(valid_elbows) if valid_elbows else None
                trial_manager.update_biomechanics(avg_elbow, trunk_lean_angle, persistent_issue_names)

                # Save frame data
                if session_collector:
                    trial_config = trial_manager.get_current_config()

                    # Build measurements dict with visibility flags
                    frame_measurements = {
                        'elbow_r': right_elbow_angle,
                        'elbow_l': left_elbow_angle,
                        'trunk_lean': trunk_lean_angle,
                        'knee_r': right_knee_angle,
                        'knee_l': left_knee_angle,
                        'step_length': step_length,
                        'step_ratio': step_ratio,
                        'base_width': base_of_support,
                        'base_ratio': base_ratio,
                        'shoulder_asym': shoulder_elevation_percent,
                        'gait_phase': current_phase,
                        'phase_confidence': phase_confidence,
                        # Visibility flags
                        'right_arm_visible': right_arm_visible,
                        'left_arm_visible': left_arm_visible,
                        'right_leg_visible': right_leg_visible,
                        'left_leg_visible': left_leg_visible,
                        'trunk_visible': trunk_visible,
                        'feet_visible': feet_visible,
                        'using_3d_landmarks': world_landmarks is not None
                    }

                    session_collector.add_frame(
                        measurements=frame_measurements,
                        evaluations=evaluations,
                        smoothed=smoothed_values,
                        persistent_issues=persistent_issue_names,
                        trial_config=trial_config
                    )

                    # Collect frame for trial analysis (only include valid measurements)
                    current_trial_frames.append({
                        'timestamp': current_time,
                        'measurements': {
                            'elbow_r': right_elbow_angle,
                            'elbow_l': left_elbow_angle,
                            'trunk_lean': trunk_lean_angle,
                            'gait_phase': current_phase,
                            'right_arm_visible': right_arm_visible,
                            'left_arm_visible': left_arm_visible,
                            'trunk_visible': trunk_visible
                        },
                        'evaluations': evaluations.copy()
                    })

                # ========================================
                # Display feedback
                # ========================================

                # Get smoothed values for display
                display_elbow_r = tracker.get_smoothed('elbow_r')
                display_elbow_l = tracker.get_smoothed('elbow_l')
                display_trunk = tracker.get_smoothed('trunk_lean')
                display_knee_r = tracker.get_smoothed('knee_r')
                display_knee_l = tracker.get_smoothed('knee_l')
                display_step = tracker.get_smoothed('step_length')
                display_base = tracker.get_smoothed('base_width')
                display_shoulder = tracker.get_smoothed('shoulder_asym')

                # Use smoothed values if available, fallback to raw, or show N/A
                elbow_r_display = display_elbow_r if display_elbow_r is not None else right_elbow_angle
                elbow_l_display = display_elbow_l if display_elbow_l is not None else left_elbow_angle
                trunk_display = display_trunk if display_trunk is not None else trunk_lean_angle
                knee_r_display = display_knee_r if display_knee_r is not None else right_knee_angle
                knee_l_display = display_knee_l if display_knee_l is not None else left_knee_angle
                step_display = display_step if display_step is not None else step_length
                base_display = display_base if display_base is not None else base_of_support
                shoulder_display = display_shoulder if display_shoulder is not None else shoulder_elevation_percent
                
                # Pre-format strings with color coding (handle None values)
                def format_val(val, suffix=""):
                    return f"{val:.1f}{suffix}" if val is not None else "N/A"

                text_data = [
                    (f"R Elbow: {format_val(elbow_r_display, ' deg')}", 30, evaluations['elbow_r']['color']),
                    (f"L Elbow: {format_val(elbow_l_display, ' deg')}", 60, evaluations['elbow_l']['color']),
                    (f"R Knee: {format_val(knee_r_display, ' deg')}", 90, evaluations['knee_r']['color']),
                    (f"L Knee: {format_val(knee_l_display, ' deg')}", 120, evaluations['knee_l']['color']),
                    (f"Trunk Lean: {format_val(trunk_display, ' deg')}", 150, evaluations['trunk']['color']),
                    (f"Step Length: {format_val(step_display, ' cm')}", 180, evaluations['step']['color']),
                    (f"Base Width: {format_val(base_display, ' cm')}", 210, evaluations['base']['color']),
                    (f"Shoulder Asym: {format_val(shoulder_display, '%')}", 240, evaluations['shoulder']['color']),
                    (f"Leg Length: {avg_leg_length:.1f} cm", 270, (200, 200, 200))
                ]
                
                # Batch render text
                for text, y_pos, color in text_data:
                    cv2.putText(image, text, (10, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Add crutch feedback messages if available
                if crutch_evaluation and crutch_evaluation['messages']:
                    for msg in crutch_evaluation['messages'][:2]:
                        if msg not in feedback_messages:
                            feedback_messages.append(msg)

                # Draw crutch detections if enabled
                if crutch_detection_enabled and crutch_detections:
                    image = crutch_detector.draw_detections(image, crutch_detections)

                    # Show crutch detection status
                    if crutch_detections['confidence'] > 0.3:
                        crutch_status_color = (0, 255, 0) if crutch_evaluation and crutch_evaluation['symmetry_status'] == 'good' else (0, 165, 255)
                        cv2.putText(image, f"Crutch: {crutch_detections['confidence']*100:.0f}%",
                                   (image_width - 120, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, crutch_status_color, 2)

                # Display top 3 feedback messages
                y_offset = 310
                for i, msg in enumerate(feedback_messages[:3]):
                    cv2.putText(image, f"! {msg}", 
                               (10, y_offset + i*35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (0, 0, 255), 2)
                
                # ========================================
                # DISPLAY GAIT PHASE
                # ========================================
                
                # Phase color coding
                phase_colors = {
                    'STANDING': (0, 255, 0),
                    'WEIGHT_BEARING_LEFT': (0, 165, 255),
                    'WEIGHT_BEARING_RIGHT': (255, 165, 0),
                    'DOUBLE_SUPPORT': (255, 255, 0),
                    'SWING_PHASE': (255, 0, 255)
                }
                
                # Phase label formatting
                phase_labels = {
                    'STANDING': 'Standing',
                    'WEIGHT_BEARING_LEFT': 'Weight: Left',
                    'WEIGHT_BEARING_RIGHT': 'Weight: Right',
                    'DOUBLE_SUPPORT': 'Double Support',
                    'SWING_PHASE': 'Swing Phase'
                }
                
                phase_color = phase_colors.get(current_phase, (200, 200, 200))
                phase_label = phase_labels.get(current_phase, current_phase)
                
                # Draw phase indicator box
                box_x, box_y = 10, image_height - 60
                box_w, box_h = 220, 50
                
                cv2.rectangle(image, (box_x, box_y), (box_x + box_w, box_y + box_h), 
                             (40, 40, 40), -1)
                cv2.rectangle(image, (box_x, box_y), (box_x + box_w, box_y + box_h), 
                             phase_color, 2)
                
                cv2.putText(image, f"Phase: {phase_label}", 
                           (box_x + 10, box_y + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, phase_color, 2)
                
                # Confidence indicator
                confidence_text = f"({phase_confidence*100:.0f}%)"
                cv2.putText(image, confidence_text, 
                           (box_x + 140, box_y + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # Draw trial overlay
                trial_manager.draw_config_overlay(image)
            
            # ========================================
            # SHOW FRAME & CHECK INPUT
            # ========================================

            # Show frame
            cv2.imshow("Mediapipe Feed", image)

            # Check if window was closed
            if cv2.getWindowProperty("Mediapipe Feed", cv2.WND_PROP_VISIBLE) < 1:
                cleanup_and_exit(cap, session_collector, trial_manager)

            # Check for exit keys
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q') or key == 27:
                # Final analysis before exit
                if len(current_trial_frames) > 0:
                    print(f"\n{'='*70}")
                    print(f"ANALYZING FINAL TRIAL...")
                    print(f"{'='*70}")

                    biomech = live_optimizer.analyze_weight_bearing_frames(current_trial_frames)
                    if biomech is not None:
                        score = live_optimizer.compute_fit_score(biomech)
                        trial_scores.append(score)
                        live_optimizer.print_analysis(biomech, score)

                # Print comprehensive fitting summary
                if len(trial_scores) > 0:
                    print(f"\n{'='*70}")
                    print(f"FITTING SESSION SUMMARY")
                    print(f"{'='*70}")

                    # Filter out invalid trials
                    valid_scores = [(i+1, s) for i, s in enumerate(trial_scores) if s != float('inf')]

                    if valid_scores:
                        print(f"\n📊 Trial History:")
                        best_score = min(s for _, s in valid_scores)
                        best_trial_idx = [i for i, s in valid_scores if s == best_score][0]

                        user_feelings_map = {1: "Good", 2: "Okay", 3: "Bad", 4: ""}

                        for trial_idx, s in valid_scores:
                            # Get user feeling if available
                            feeling_str = ""
                            if hasattr(trial_manager, 'trial_feelings') and trial_idx in trial_manager.trial_feelings:
                                f = trial_manager.trial_feelings[trial_idx]
                                if f < 4:
                                    feeling_str = f" (felt: {user_feelings_map[f]})"

                            # Get config if available
                            config_str = ""
                            if hasattr(trial_manager, 'trial_configs') and trial_idx in trial_manager.trial_configs:
                                cfg = trial_manager.trial_configs[trial_idx]['config']
                                g_idx = cfg.get('grip_setting_idx', 0)
                                o_idx = cfg.get('overall_setting_idx', 0)
                                config_str = f" [Grip #{g_idx+1}, Overall #{o_idx+1}]"

                            marker = " ⭐ BEST" if s == best_score else ""
                            print(f"   Trial {trial_idx}: Score {s:.2f}{feeling_str}{config_str}{marker}")

                        if len(valid_scores) > 1:
                            first_score = valid_scores[0][1]
                            total_improvement = first_score - best_score
                            print(f"\n   Total improvement: {total_improvement:.2f}")

                            if total_improvement > 2.0:
                                print(f"   🎉 Excellent progress!")
                            elif total_improvement > 1.0:
                                print(f"   ✅ Good improvement")
                            elif total_improvement > 0.5:
                                print(f"   👍 Some improvement")
                            else:
                                print(f"   ℹ️  Minimal change - initial settings were already good!")

                        # Show recommended final settings
                        print(f"\n💡 RECOMMENDED FINAL SETTINGS (from Trial {best_trial_idx}):")
                        if hasattr(trial_manager, 'trial_configs') and best_trial_idx in trial_manager.trial_configs:
                            best_cfg = trial_manager.trial_configs[best_trial_idx]['config']
                            g_idx = best_cfg.get('grip_setting_idx', 0)
                            o_idx = best_cfg.get('overall_setting_idx', 0)
                            g_cm = best_cfg.get('grip_height_cm', device_profile.setting_to_cm(g_idx, 'grip'))
                            o_cm = best_cfg.get('overall_length_cm', device_profile.setting_to_cm(o_idx, 'overall'))
                            print(f"   Grip: Hole #{g_idx+1} ({g_cm:.1f}cm / {g_cm/2.54:.1f}\")")
                            print(f"   Overall: Position #{o_idx+1} ({o_cm:.1f}cm / {o_cm/2.54:.1f}\")")
                        else:
                            print(f"   Score: {best_score:.2f}")

                        # Check if user feeling matched score
                        if hasattr(trial_manager, 'trial_feelings'):
                            best_feeling = trial_manager.trial_feelings.get(best_trial_idx, 4)
                            if best_feeling == 1:
                                print(f"\n   ✅ This also felt the best to you - great match!")
                            elif best_feeling == 3:
                                # Find trial that felt best
                                good_trials = [t for t, f in trial_manager.trial_feelings.items() if f == 1]
                                if good_trials:
                                    print(f"\n   ⚠️  Note: You said Trial {good_trials[0]} felt better.")
                                    print(f"      Consider those settings if comfort is priority.")

                    print(f"\n{'='*70}\n")

                cleanup_and_exit(cap, session_collector, trial_manager)
            elif key == ord('['):  # Decrease grip
                trial_manager.adjust_grip(-1)
            elif key == ord(']'):  # Increase grip
                trial_manager.adjust_grip(+1)
            elif key == ord('-') or key == ord('_'):  # Decrease overall
                trial_manager.adjust_overall(-1)
            elif key == ord('=') or key == ord('+'):  # Increase overall
                trial_manager.adjust_overall(+1)
            elif key == ord('c') or key == ord('C'):  # Toggle crutch detection
                crutch_detection_enabled = not crutch_detection_enabled
                status = "ON" if crutch_detection_enabled else "OFF"
                print(f"Crutch detection: {status}")
                if not crutch_detection_enabled:
                    crutch_detector.reset()
            elif key == ord('n') or key == ord('N'):  # New trial - analyze and recommend
                # Only analyze if we have frames from previous trial
                if len(current_trial_frames) > 0:
                    trial_num = trial_manager.current_trial_index
                    print(f"\n{'='*70}")
                    print(f"TRIAL {trial_num} COMPLETE - FEEDBACK & ANALYSIS")
                    print(f"{'='*70}")

                    # ========================================
                    # 1. ASK USER HOW THEY FELT
                    # ========================================
                    print(f"\n📋 How did that trial feel?")
                    print(f"   1. Good - comfortable, stable")
                    print(f"   2. Okay - some minor discomfort")
                    print(f"   3. Bad - uncomfortable, unstable")
                    print(f"   4. Skip feedback")

                    try:
                        feeling = input("\n   Your rating (1-4): ").strip()
                        feeling_score = int(feeling) if feeling in ['1', '2', '3', '4'] else 4
                    except:
                        feeling_score = 4

                    user_feelings = {1: "Good", 2: "Okay", 3: "Bad", 4: "Skipped"}
                    if feeling_score < 4:
                        print(f"   ✓ Recorded: {user_feelings[feeling_score]}")

                    # Store user feeling with trial
                    if not hasattr(trial_manager, 'trial_feelings'):
                        trial_manager.trial_feelings = {}
                    trial_manager.trial_feelings[trial_num] = feeling_score

                    # ========================================
                    # 2. ANALYZE BIOMECHANICS
                    # ========================================
                    biomech = live_optimizer.analyze_weight_bearing_frames(current_trial_frames)

                    if biomech is not None:
                        # Calculate score
                        score = live_optimizer.compute_fit_score(biomech)
                        trial_scores.append(score)

                        # Store config with score for comparison
                        current_config = trial_manager.get_current_config()
                        if not hasattr(trial_manager, 'trial_configs'):
                            trial_manager.trial_configs = {}
                        trial_manager.trial_configs[trial_num] = {
                            'config': current_config.copy(),
                            'score': score,
                            'feeling': feeling_score,
                            'biomech': biomech.copy()
                        }

                        # Display analysis
                        live_optimizer.print_analysis(biomech, score)

                        # ========================================
                        # 3. COMPARE WITH PREVIOUS TRIALS
                        # ========================================
                        if len(trial_scores) > 1:
                            print(f"\n📊 TRIAL COMPARISON:")
                            best_score = min(trial_scores)
                            best_trial = trial_scores.index(best_score) + 1

                            for i, s in enumerate(trial_scores, 1):
                                feeling_str = ""
                                if hasattr(trial_manager, 'trial_feelings') and i in trial_manager.trial_feelings:
                                    f = trial_manager.trial_feelings[i]
                                    if f < 4:
                                        feeling_str = f" (felt: {user_feelings[f]})"

                                if i == len(trial_scores):  # Current trial
                                    marker = " ← CURRENT"
                                elif s == best_score:
                                    marker = " ⭐ BEST SO FAR"
                                else:
                                    marker = ""
                                print(f"   Trial {i}: {s:.2f}{feeling_str}{marker}")

                            # Hint about best trial
                            if trial_scores[-1] > best_score + 0.5:
                                print(f"\n   💡 HINT: Trial {best_trial} had a better score ({best_score:.2f})")
                                print(f"      Consider going back to those settings!")

                                # Show what the best settings were
                                if hasattr(trial_manager, 'trial_configs') and best_trial in trial_manager.trial_configs:
                                    best_cfg = trial_manager.trial_configs[best_trial]['config']
                                    print(f"      Best was: Grip #{best_cfg.get('grip_setting_idx', 0)+1}, Overall #{best_cfg.get('overall_setting_idx', 0)+1}")

                        # ========================================
                        # 4. GET AI RECOMMENDATIONS
                        # ========================================
                        suggestions = live_optimizer.recommend_adjustment(biomech, current_config, device_profile)

                        # Check if optimal
                        is_optimal = any(s['type'] == 'success' for s in suggestions)

                        if is_optimal:
                            print(f"\n🎉 OPTIMAL FIT ACHIEVED!")
                            print(f"   Your biomechanics look great!")

                            # Check user feeling matches
                            if feeling_score == 1:
                                print(f"   ✅ And you said it felt good - perfect match!")
                                print(f"\n   Press 'q' to finish, or continue testing.")
                            elif feeling_score == 3:
                                print(f"   ⚠️  But you said it felt bad - let's keep adjusting.")
                        else:
                            live_optimizer.print_suggestions(suggestions)

                            # ========================================
                            # 5. RECOMMEND NEW SETTINGS
                            # ========================================
                            actionable = [s for s in suggestions if s['type'] not in ['success', 'warning']]

                            if actionable:
                                print(f"\n🔧 RECOMMENDED NEW SETTINGS:")

                                # Apply the first actionable suggestion
                                sug = actionable[0]
                                new_config = live_optimizer.apply_suggestion(sug, current_config, device_profile)

                                new_grip_idx = new_config.get('grip_setting_idx', current_config.get('grip_setting_idx', 0))
                                new_overall_idx = new_config.get('overall_setting_idx', current_config.get('overall_setting_idx', 0))

                                print(f"   Grip: Hole #{new_grip_idx + 1} ({new_config.get('grip_height_cm', 0):.1f}cm)")
                                print(f"   Overall: Position #{new_overall_idx + 1} ({new_config.get('overall_length_cm', 0):.1f}cm)")

                                print(f"\n   Options:")
                                print(f"   [Enter] Apply recommendation and start next trial")
                                print(f"   [s] Skip - keep current settings")
                                print(f"   [m] Manual - enter your own settings")

                                choice = input("\n   Your choice: ").strip().lower()

                                if choice == 's':
                                    print("   ✓ Keeping current settings")
                                elif choice == 'm':
                                    # Manual adjustment
                                    try:
                                        print(f"\n   Enter new settings (1-{device_profile.grip_num_settings} for grip, 1-{device_profile.overall_num_settings} for overall):")
                                        new_grip = int(input(f"   Grip hole #: ").strip()) - 1
                                        new_overall = int(input(f"   Overall position #: ").strip()) - 1

                                        # Validate and apply
                                        if 0 <= new_grip < device_profile.grip_num_settings:
                                            trial_manager.current_grip_idx = new_grip
                                        if 0 <= new_overall < device_profile.overall_num_settings:
                                            trial_manager.current_overall_idx = new_overall

                                        print(f"   ✓ Settings updated")
                                    except:
                                        print(f"   ✓ Keeping current settings")
                                else:
                                    # Apply the recommendation
                                    trial_manager.current_grip_idx = new_grip_idx
                                    trial_manager.current_overall_idx = new_overall_idx
                                    print(f"   ✓ New settings applied!")

                        print(f"\n   Adjust your crutches to the recommended settings.")
                        print(f"   Then walk for another 20-30 seconds.")

                    else:
                        print(f"\n⚠️  Not enough weight-bearing data in this trial.")
                        print(f"   Try walking longer (at least 20 seconds).")
                        trial_scores.append(float('inf'))  # Mark as invalid

                    print(f"\n{'='*70}")
                    input("Press Enter when ready for next trial...")
                    print(f"\n")

                # Start new trial
                trial_manager.start_new_trial(collect_feedback=False)  # We already collected feedback

                # Reset frame collection for new trial
                current_trial_frames = []
            elif key == ord('e') or key == ord('E'):  # End trial and collect feedback only
                trial_manager.end_trial_with_feedback()
            elif key == ord('b') or key == ord('B'):  # Mark best
                trial_manager.mark_as_best()

    # Fallback cleanup
    cleanup_and_exit(cap, session_collector, trial_manager)


if __name__ == "__main__":
    print("="*70)
    print("CRUTCH GAIT ANALYSIS")
    print("="*70)
    main()