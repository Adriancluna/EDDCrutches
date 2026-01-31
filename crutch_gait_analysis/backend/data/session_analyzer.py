"""
Cell 4.6: Session Analysis & Reporting
Smart analysis that ignores noise and finds real patterns
"""

import json
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime

class SessionAnalyzer:
    """Analyze session data with noise filtering and pattern detection"""
    
    def __init__(self, session_filepath):
        """Load session from JSON file"""
        with open(session_filepath, 'r') as f:
            self.session = json.load(f)
        
        self.metadata = self.session['metadata']
        self.frames = self.session['frames']
        self.total_frames = len(self.frames)
        
        # Analysis results (computed on demand)
        self._issue_clusters = None
        self._measurement_stats = None
        
    def get_measurement_stats(self):
        """Calculate statistics for all measurements"""
        if self._measurement_stats is not None:
            return self._measurement_stats
        
        stats = {}
        
        # Get all measurement names from first frame
        measurement_names = self.frames[0]['measurements'].keys()
        
        for name in measurement_names:
            values = [f['measurements'][name] for f in self.frames]
            smoothed_values = [f['smoothed_values'][name] for f in self.frames 
                              if f['smoothed_values'][name] is not None]
            
            stats[name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'smoothed_mean': np.mean(smoothed_values) if smoothed_values else None,
                'range': np.max(values) - np.min(values)
            }
        
        self._measurement_stats = stats
        return stats
    
    def find_issue_clusters(self, min_cluster_size=10):
        """
        Find clusters of persistent issues (ignores isolated glitches).
        
        A cluster = consecutive frames where the same issue appears.
        Only clusters >= min_cluster_size are considered "real issues".
        """
        if self._issue_clusters is not None:
            return self._issue_clusters
        
        clusters = defaultdict(list)  # {issue_name: [cluster1, cluster2, ...]}
        
        # Get all possible issue types
        issue_types = set()
        for frame in self.frames:
            issue_types.update(frame['persistent_issues'])
        
        # Find clusters for each issue type
        for issue_type in issue_types:
            current_cluster = []
            
            for i, frame in enumerate(self.frames):
                if issue_type in frame['persistent_issues']:
                    # Issue present - add to current cluster
                    current_cluster.append(i)
                else:
                    # Issue absent - save cluster if big enough
                    if len(current_cluster) >= min_cluster_size:
                        clusters[issue_type].append({
                            'start_frame': current_cluster[0],
                            'end_frame': current_cluster[-1],
                            'duration_frames': len(current_cluster),
                            'start_time': self.frames[current_cluster[0]]['timestamp'],
                            'end_time': self.frames[current_cluster[-1]]['timestamp']
                        })
                    current_cluster = []
            
            # Don't forget final cluster
            if len(current_cluster) >= min_cluster_size:
                clusters[issue_type].append({
                    'start_frame': current_cluster[0],
                    'end_frame': current_cluster[-1],
                    'duration_frames': len(current_cluster),
                    'start_time': self.frames[current_cluster[0]]['timestamp'],
                    'end_time': self.frames[current_cluster[-1]]['timestamp']
                })
        
        self._issue_clusters = dict(clusters)
        return self._issue_clusters
    
    def get_issue_severity_score(self):
        """
        Calculate overall severity score (0-100).
        
        0 = Perfect form throughout
        100 = Constant critical issues
        """
        clusters = self.find_issue_clusters()
        
        if not clusters:
            return 0
        
        # Count total frames with clustered issues
        total_problem_frames = sum(
            sum(c['duration_frames'] for c in cluster_list)
            for cluster_list in clusters.values()
        )
        
        # Percentage of session with problems
        percentage = (total_problem_frames / self.total_frames) * 100
        
        return min(100, percentage)
    
    def get_top_issues(self, top_n=3):
        """
        Get the most significant issues (by total duration in clusters).
        
        Returns: List of (issue_name, total_frames, percentage) sorted by severity
        """
        clusters = self.find_issue_clusters()
        
        issue_durations = []
        for issue_name, cluster_list in clusters.items():
            total_frames = sum(c['duration_frames'] for c in cluster_list)
            percentage = (total_frames / self.total_frames) * 100
            issue_durations.append((issue_name, total_frames, percentage))
        
        # Sort by duration (descending)
        issue_durations.sort(key=lambda x: x[1], reverse=True)
        
        return issue_durations[:top_n]
    
    def get_fatigue_analysis(self):
        """
        Detect if issues worsen over time (fatigue indicator).
        
        Splits session into thirds and compares issue frequency.
        """
        third = self.total_frames // 3
        
        # Split into early, middle, late
        early = self.frames[:third]
        middle = self.frames[third:2*third]
        late = self.frames[2*third:]
        
        def count_issues(frame_list):
            total = 0
            for frame in frame_list:
                total += len(frame['persistent_issues'])
            return total / len(frame_list) if frame_list else 0
        
        early_rate = count_issues(early)
        middle_rate = count_issues(middle)
        late_rate = count_issues(late)
        
        # Detect trend
        if late_rate > early_rate * 1.5:
            trend = "worsening"
        elif late_rate < early_rate * 0.7:
            trend = "improving"
        else:
            trend = "stable"
        
        return {
            'trend': trend,
            'early_rate': early_rate,
            'middle_rate': middle_rate,
            'late_rate': late_rate
        }
    
    def print_summary_report(self):
        """Print a comprehensive, human-readable session summary"""
        
        print("\n" + "="*70)
        print("SESSION ANALYSIS REPORT")
        print("="*70)
        
        # Header
        print(f"\n📅 Session: {self.metadata['session_id']}")
        print(f"⏱️  Duration: {self.metadata['duration_seconds']:.1f} seconds")
        print(f"🎬 Total Frames: {self.total_frames}")
        print(f"👤 User Height: {self.metadata['user_height_cm']:.1f} cm")
        
        # Overall Score
        severity = self.get_issue_severity_score()
        if severity < 10:
            grade = "A (Excellent)"
            emoji = "🟢"
        elif severity < 25:
            grade = "B (Good)"
            emoji = "🟡"
        elif severity < 50:
            grade = "C (Needs Improvement)"
            emoji = "🟠"
        else:
            grade = "D (Poor Form)"
            emoji = "🔴"
        
        print(f"\n{emoji} Overall Form Grade: {grade}")
        print(f"   Issue Severity: {severity:.1f}%")
        
        # Top Issues
        print(f"\n🎯 TOP ISSUES (Persistent Patterns Only):")
        top_issues = self.get_top_issues(top_n=5)
        
        if not top_issues:
            print("   ✅ No persistent issues detected!")
            print("   Great job maintaining proper form!")
        else:
            for i, (issue_name, frames, percentage) in enumerate(top_issues, 1):
                issue_label = {
                    'elbow_r': 'Right Elbow',
                    'elbow_l': 'Left Elbow',
                    'trunk': 'Trunk Lean',
                    'knee_r': 'Right Knee',
                    'knee_l': 'Left Knee',
                    'step': 'Step Length',
                    'base': 'Base Width',
                    'shoulder': 'Shoulder Asymmetry'
                }.get(issue_name, issue_name)
                
                print(f"   {i}. {issue_label}: {frames} frames ({percentage:.1f}%)")
        
        # Issue Timeline
        clusters = self.find_issue_clusters()
        if clusters:
            print(f"\n📊 ISSUE TIMELINE:")
            for issue_name, cluster_list in sorted(clusters.items(), 
                                                   key=lambda x: sum(c['duration_frames'] 
                                                                    for c in x[1]), 
                                                   reverse=True)[:3]:
                issue_label = {
                    'elbow_r': 'Right Elbow',
                    'elbow_l': 'Left Elbow',
                    'trunk': 'Trunk Lean',
                    'knee_r': 'Right Knee',
                    'knee_l': 'Left Knee',
                    'step': 'Step Length',
                    'base': 'Base Width',
                    'shoulder': 'Shoulder Asymmetry'
                }.get(issue_name, issue_name)
                
                print(f"\n   {issue_label}:")
                for j, cluster in enumerate(cluster_list[:3], 1):  # Show max 3 clusters
                    duration = cluster['end_time'] - cluster['start_time']
                    print(f"     • {cluster['start_time']:.1f}s - {cluster['end_time']:.1f}s "
                          f"({duration:.1f}s, {cluster['duration_frames']} frames)")
        
        # Fatigue Analysis
        fatigue = self.get_fatigue_analysis()
        print(f"\n💪 FATIGUE ANALYSIS:")
        print(f"   Trend: {fatigue['trend'].upper()}")
        print(f"   Early session issues: {fatigue['early_rate']:.2f} per frame")
        print(f"   Late session issues: {fatigue['late_rate']:.2f} per frame")
        
        if fatigue['trend'] == 'worsening':
            print("   ⚠️  Form deteriorated over time - consider shorter sessions")
        elif fatigue['trend'] == 'improving':
            print("   ✅ Form improved over time - good adaptation!")
        else:
            print("   ✅ Consistent form throughout session")
        
        # Measurement Statistics
        stats = self.get_measurement_stats()
        print(f"\n📏 KEY MEASUREMENTS (Averages):")
        
        measurement_labels = {
            'elbow_r': ('Right Elbow', 'deg'),
            'elbow_l': ('Left Elbow', 'deg'),
            'trunk_lean': ('Trunk Lean', 'deg'),
            'knee_r': ('Right Knee', 'deg'),
            'knee_l': ('Left Knee', 'deg'),
            'step_length': ('Step Length', 'cm'),
            'base_width': ('Base Width', 'cm'),
            'shoulder_asym': ('Shoulder Asym', '%')
        }
        
        for name, (label, unit) in measurement_labels.items():
            if name in stats:
                mean = stats[name]['smoothed_mean'] or stats[name]['mean']
                std = stats[name]['std']
                print(f"   {label:18s}: {mean:6.1f} {unit} (±{std:.1f})")
        
        print("\n" + "="*70)
        print("END OF REPORT")
        print("="*70 + "\n")
    
    def export_summary_json(self, output_filepath=None):
        """Export analysis summary as JSON for further processing"""
        
        if output_filepath is None:
            output_filepath = f"analysis_{self.metadata['session_id']}.json"
        
        summary = {
            'session_id': self.metadata['session_id'],
            'analyzed_at': datetime.now().isoformat(),
            'severity_score': self.get_issue_severity_score(),
            'top_issues': [
                {'name': name, 'frames': frames, 'percentage': pct}
                for name, frames, pct in self.get_top_issues(top_n=5)
            ],
            'issue_clusters': self.find_issue_clusters(),
            'fatigue_analysis': self.get_fatigue_analysis(),
            'measurement_stats': self.get_measurement_stats()
        }
        
        with open(output_filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Analysis exported to: {output_filepath}")
        return output_filepath

print("✓ Session analyzer loaded")