"""Analysis module for extracting workflows from recordings."""

from .schema import (
    # Core workflow types
    Parameter,
    ParameterType,
    Workflow,
    # Multi-pass extraction types
    DetectedEvent,
    DetectedParameter,
    ExtractionProgress,
    ProgressCallback,
    RunningUnderstanding,
    WorkflowStep,
    # Legacy types
    RecordedEvent,
    RecordedSession,
)
from .workflow_extractor import WorkflowExtractor
from .parameter_detector import ParameterDetector

__all__ = [
    # Core workflow types
    "Parameter",
    "ParameterType",
    "Workflow",
    # Multi-pass extraction types
    "DetectedEvent",
    "DetectedParameter",
    "ExtractionProgress",
    "ProgressCallback",
    "RunningUnderstanding",
    "WorkflowStep",
    # Extractors
    "WorkflowExtractor",
    "ParameterDetector",
    # Legacy types
    "RecordedEvent",
    "RecordedSession",
]
