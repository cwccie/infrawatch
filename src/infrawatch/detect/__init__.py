"""Detection pipeline.

Configurable thresholds, multi-metric correlation, context-aware detection,
and severity classification.
"""

from infrawatch.detect.pipeline import DetectionPipeline, DetectionResult
from infrawatch.detect.severity import Severity, classify_severity
from infrawatch.detect.context import ContextAnalyzer

__all__ = [
    "DetectionPipeline",
    "DetectionResult",
    "Severity",
    "classify_severity",
    "ContextAnalyzer",
]
