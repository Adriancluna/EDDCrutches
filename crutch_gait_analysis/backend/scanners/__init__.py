"""
Crutch Scanners Module

Available scanners:
- simple_scanner: Measures crutch length using edge detection
- hole_detector: Single-frame hole detection (quick scan)
- hole_scanner: Multi-frame hole detection with guided workflow (recommended)
"""

from .simple_scanner import scan_crutch_simple
from .hole_detector import scan_crutch_holes as scan_crutch_holes_quick
from .hole_scanner import scan_crutch_holes

__all__ = [
    'scan_crutch_simple',
    'scan_crutch_holes',
    'scan_crutch_holes_quick'
]
