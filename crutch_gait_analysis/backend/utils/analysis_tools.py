"""
Analysis helper functions
"""
import json
import glob
import os
from data.session_analyzer import SessionAnalyzer


def analyze_session(session_filepath):
    """
    Quick function to analyze a session and print report.
    
    Usage:
        analyze_session('sessions/session_20260127_143022.json')
    """
    analyzer = SessionAnalyzer(session_filepath)
    analyzer.print_summary_report()
    return analyzer


def analyze_latest_session():
    """Analyze the most recent session automatically"""
    
    # Find all session files
    session_files = glob.glob('sessions/session_*.json')
    
    if not session_files:
        print("❌ No session files found in sessions/ directory")
        return None
    
    # Get the most recent one
    latest = max(session_files, key=os.path.getmtime)
    print(f"📂 Analyzing: {latest}\n")
    
    return analyze_session(latest)


def analyze_phase_distribution(session_filepath):
    """Analyze how time was spent in each phase"""
    
    with open(session_filepath, 'r') as f:
        session = json.load(f)
    
    frames = session['frames']
    total_frames = len(frames)
    
    # Count frames per phase
    phase_counts = {}
    for frame in frames:
        phase = frame['measurements'].get('gait_phase', 'UNKNOWN')
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    
    print("\n" + "="*60)
    print("GAIT PHASE DISTRIBUTION")
    print("="*60)
    
    phase_labels = {
        'STANDING': 'Standing',
        'WEIGHT_BEARING_LEFT': 'Weight-Bearing (Left)',
        'WEIGHT_BEARING_RIGHT': 'Weight-Bearing (Right)',
        'DOUBLE_SUPPORT': 'Double Support',
        'SWING_PHASE': 'Swing Phase'
    }
    
    for phase, count in sorted(phase_counts.items(), key=lambda x: -x[1]):
        percentage = (count / total_frames) * 100
        label = phase_labels.get(phase, phase)
        
        # Visual bar
        bar_length = int(percentage / 2)  # Scale to fit console
        bar = "█" * bar_length
        
        print(f"{label:25s} │ {bar:50s} {percentage:5.1f}% ({count} frames)")
    
    print("="*60 + "\n")