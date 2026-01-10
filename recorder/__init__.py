"""Recording module for workflow automation.

This module provides:
- RecordingSession: Record screen using macOS screencapture
- VideoProcessor: Process video files to extract frames and audio
- Transcriber: Transcribe audio using OpenAI Speech-to-Text API
"""

from .session import RecordingSession, record_video
from .video_processor import VideoProcessor, ProcessedSession, FrameInfo
from .transcriber import Transcriber, Transcript, TranscriptSegment, TranscriptWord

__all__ = [
    "RecordingSession",
    "record_video",
    "VideoProcessor",
    "ProcessedSession",
    "FrameInfo",
    "Transcriber",
    "Transcript",
    "TranscriptSegment",
    "TranscriptWord",
]
