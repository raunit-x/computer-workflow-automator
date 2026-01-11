"""Workflow extraction from video recordings using LLM providers."""

import base64
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image
from tqdm import tqdm

from .schema import (
    DetectedEvent,
    RunningUnderstanding,
    Workflow,
)
from .json_utils import extract_json_from_response
from prompts.analyzer_prompts import (
    EVENT_DETECTION_PROMPT,
    UNDERSTANDING_UPDATE_PROMPT,
    WORKFLOW_SYNTHESIS_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    from recorder.video_processor import ProcessedSession
    from utils.logger import WorkflowLogger
    from utils.llm import LLMClient
    from config import ModelConfig


class WorkflowExtractor:
    """Extracts structured workflows from video recordings using LLM providers.
    
    Uses a unified LLMClient that supports Anthropic, OpenAI, and Gemini,
    with configurable models for each extraction stage.
    """
    
    # Maximum image dimension for API requests (to avoid request size limits)
    MAX_IMAGE_DIMENSION = 1280
    # JPEG quality for compressed images
    IMAGE_QUALITY = 85
    
    def __init__(
        self,
        model_config: "ModelConfig",
        llm_client: "LLMClient",
        max_image_dimension: int | None = None,
        logger: "WorkflowLogger | None" = None,
    ):
        """Initialize the extractor.
        
        Args:
            model_config: ModelConfig with stage-specific model settings.
            llm_client: Unified LLMClient for API calls.
            max_image_dimension: Maximum dimension for resized images (default 1280).
            logger: Optional WorkflowLogger for structured output.
        """
        self.model_config = model_config
        self.llm = llm_client
        self.max_image_dimension = max_image_dimension or self.MAX_IMAGE_DIMENSION
        self.logger = logger
    
    def _log(self, message: str, level: str = "info") -> None:
        """Log a message using the logger or print."""
        if self.logger:
            if level == "info":
                self.logger.info(message)
            elif level == "step":
                self.logger.step(message)
            elif level == "success":
                self.logger.success(message)
            elif level == "warning":
                self.logger.warning(message)
            elif level == "error":
                self.logger.error(message)
            elif level == "header":
                self.logger.header(message)
        else:
            print(message)
    
    def _resize_and_encode_image(self, image_path: Path) -> tuple[str, str]:
        """Resize image if needed and return base64-encoded data.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Tuple of (base64_data, media_type).
        """
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (for JPEG encoding)
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Check if resizing is needed
            width, height = img.size
            max_dim = max(width, height)
            
            if max_dim > self.max_image_dimension:
                # Calculate new dimensions maintaining aspect ratio
                scale = self.max_image_dimension / max_dim
                new_width = int(width * scale)
                new_height = int(height * scale)
                # Use LANCZOS resampling (Resampling.LANCZOS in newer Pillow versions)
                try:
                    resample = Image.Resampling.LANCZOS
                except AttributeError:
                    resample = Image.LANCZOS  # type: ignore[attr-defined]
                img = img.resize((new_width, new_height), resample)
            
            # Encode as JPEG for smaller size
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=self.IMAGE_QUALITY, optimize=True)
            buffer.seek(0)
            
            image_data = base64.b64encode(buffer.read()).decode()
            return image_data, "image/jpeg"
    
    def extract_from_video(
        self,
        video_path: Path,
        output_dir: Path | None = None,
        max_frames: int | None = None,  # Deprecated, kept for compatibility
        openai_api_key: str | None = None,
        chunk_size: int = 10,
        verbose: bool = True,
    ) -> Workflow:
        """Extract a workflow from a video file.
        
        This is the main entry point for video-based workflow extraction.
        It processes the video to extract frames and audio, then uses
        multi-pass analysis to build a comprehensive workflow.
        
        Args:
            video_path: Path to the video file (.mov, .mp4, etc.).
            output_dir: Directory to store processed data. If None, uses temp dir.
            max_frames: Deprecated. The multi-pass approach processes all frames.
            openai_api_key: OpenAI API key for audio transcription.
            chunk_size: Number of frames to analyze per chunk in Pass 1.
            verbose: Whether to print progress information with tqdm bars.
            
        Returns:
            Extracted Workflow object.
        """
        # Import here to avoid circular imports
        from recorder.video_processor import VideoProcessor
        
        # Process the video
        processor = VideoProcessor(
            fps=2.0,  # Extract 2 frames per second
            openai_api_key=openai_api_key,
        )
        
        session = processor.process(
            video_path=video_path,
            output_dir=output_dir,
            extract_audio=True,
            transcribe=True,
        )
        
        return self.extract_from_processed_session(
            session,
            chunk_size=chunk_size,
            verbose=verbose,
        )
    
    def extract_from_processed_session(
        self,
        session: "ProcessedSession",
        max_frames: int | None = None,  # Deprecated, kept for compatibility
        chunk_size: int = 10,
        verbose: bool = True,
    ) -> Workflow:
        """Extract a workflow from a processed video session using multi-pass analysis.
        
        This method uses a three-pass approach:
        1. Event Detection: Analyze frames in chunks to detect discrete user actions
        2. Understanding Building: Incrementally build workflow understanding from events
        3. Workflow Synthesis: Generate polished markdown workflow document
        
        Args:
            session: ProcessedSession from VideoProcessor.
            max_frames: Deprecated. The multi-pass approach processes all frames.
            chunk_size: Number of frames to analyze per chunk in Pass 1.
            verbose: Whether to print progress information with tqdm bars.
            
        Returns:
            Extracted Workflow object.
        """
        frames = session.frames
        
        if verbose:
            self._log("MULTI-PASS WORKFLOW EXTRACTION", level="header")
            self._log(f"Session ID: {session.session_id}")
            self._log(f"Duration: {session.duration:.1f}s")
            self._log(f"Total frames: {len(frames)}")
            if session.transcript:
                self._log(f"Transcript: {len(session.transcript.segments)} segments")
        
        if not frames:
            # Return empty workflow if no frames
            return Workflow(
                id="empty_workflow",
                name="Empty Workflow",
                description="No frames were provided for analysis",
                parameters=[],
                instructions="No content available.",
                source_session_id=session.session_id,
            )
        
        # Pass 1: Detect events from all frames
        events = self._detect_events_pass(
            frames=frames,
            transcript=session.transcript,
            chunk_size=chunk_size,
            overlap=2,
            verbose=verbose,
        )
        
        # Pass 2: Build running understanding from events
        understanding = self._build_understanding_pass(
            events=events,
            transcript=session.transcript,
            batch_size=15,
            verbose=verbose,
        )
        
        # Pass 3: Generate final workflow
        workflow = self._generate_workflow_pass(
            understanding=understanding,
            verbose=verbose,
        )
        
        if verbose:
            self._log("EXTRACTION COMPLETE", level="header")
            self._log(f"Generated workflow: {workflow.name}", level="success")
            self._log(f"  - {len(workflow.parameters)} parameters")
            self._log(f"  - {len(workflow.instructions)} chars of instructions")
        
        workflow.source_session_id = session.session_id
        return workflow
    
    def extract_from_processed_session_legacy(
        self,
        session: "ProcessedSession",
        max_frames: int = 30,
    ) -> Workflow:
        """Legacy single-pass extraction - DEPRECATED.
        
        This method is deprecated. Use extract_from_processed_session() instead,
        which provides better results with the multi-pass approach.
        
        Args:
            session: ProcessedSession from VideoProcessor.
            max_frames: Maximum number of frames to analyze.
            
        Returns:
            Extracted Workflow object.
        """
        import warnings
        warnings.warn(
            "extract_from_processed_session_legacy is deprecated. "
            "Use extract_from_processed_session() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Redirect to the new method
        return self.extract_from_processed_session(
            session=session,
            verbose=False,
        )
    
    def extract_from_session(
        self,
        session_dir: Path,
        max_screenshots: int = 20,
        chunk_size: int = 10,
        verbose: bool = True,
    ) -> Workflow:
        """Extract a workflow from a legacy recorded session directory.
        
        This method is kept for backward compatibility with the old recording format.
        For new processed sessions, it uses the multi-pass extraction approach.
        
        Args:
            session_dir: Path to the session directory with session.json.
            max_screenshots: Maximum number of screenshots for legacy format.
            chunk_size: Number of frames per chunk for multi-pass analysis.
            verbose: Whether to print progress information with tqdm bars.
            
        Returns:
            Extracted Workflow object.
        """
        # Check if this is a new processed session or legacy format
        processed_json = session_dir / "processed_session.json"
        session_json = session_dir / "session.json"
        
        if processed_json.exists():
            # New format - load ProcessedSession and use multi-pass extraction
            from recorder.video_processor import ProcessedSession
            session = ProcessedSession.load(session_dir)
            return self.extract_from_processed_session(
                session,
                chunk_size=chunk_size,
                verbose=verbose,
            )
        
        if not session_json.exists():
            raise FileNotFoundError(f"No session data found in {session_dir}")
        
        # Legacy format
        with open(session_json) as f:
            session_data = json.load(f)
        
        # Get screenshot paths
        screenshots_dir = session_dir / "screenshots"
        screenshot_paths = sorted(screenshots_dir.glob("*.png")) if screenshots_dir.exists() else []
        
        # Also check for frames directory (new format)
        frames_dir = session_dir / "frames"
        if frames_dir.exists():
            screenshot_paths = sorted(frames_dir.glob("*.png"))
        
        # Sample screenshots if too many
        if len(screenshot_paths) > max_screenshots:
            step = len(screenshot_paths) // max_screenshots
            screenshot_paths = screenshot_paths[::step][:max_screenshots]
        
        # Build unified content format for legacy extraction
        content = self._build_legacy_extraction_content(
            session_data=session_data,
            screenshot_paths=screenshot_paths,
        )
        
        # Use LLMClient with synthesis model for legacy extraction
        response_text = self.llm.generate(
            model=self.model_config.synthesis,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            content=content,
            max_tokens=8192,
            phase="legacy_extraction",
        )
        
        # Parse the markdown response
        markdown_content = self._extract_markdown(response_text)
        
        # Create workflow from markdown
        workflow = Workflow.from_markdown(markdown_content)
        workflow.source_session_id = session_data.get("session_id")
        
        return workflow
    
    def _build_legacy_extraction_content(
        self,
        session_data: dict,
        screenshot_paths: list[Path],
    ) -> list[dict]:
        """Build unified content format for legacy session extraction."""
        content = []
        
        # Add text description
        text_parts = ["# Screen Recording Analysis\n\n"]
        text_parts.append("This is a screen recording of a workflow demonstration.\n\n")
        
        # Add metadata from session
        if "duration" in session_data:
            text_parts.append(f"Duration: {session_data['duration']:.1f} seconds\n")
        if "click_count" in session_data:
            text_parts.append(f"Mouse clicks: {session_data['click_count']}\n")
        if "keystroke_count" in session_data:
            text_parts.append(f"Keystrokes: {session_data['keystroke_count']}\n")
        
        text_parts.append(f"\nNumber of screenshots: {len(screenshot_paths)}\n")
        
        # Add transcript if available
        if session_data.get("transcript"):
            text_parts.append("\n## Voice-Over Transcript\n\n")
            text_parts.append(session_data["transcript"])
        elif session_data.get("audio_transcript"):
            text_parts.append("\n## Voice-Over Transcript\n\n")
            text_parts.append(session_data["audio_transcript"])
        
        content.append({"type": "text", "text": "".join(text_parts)})
        
        # Add screenshots in unified format
        content.append({
            "type": "text",
            "text": f"\n## Screenshots ({len(screenshot_paths)} images)\n\n"
        })
        
        for i, screenshot_path in enumerate(screenshot_paths):
            # Resize and encode image
            image_data, media_type = self._resize_and_encode_image(screenshot_path)
            
            content.append({
                "type": "image",
                "data": image_data,
                "media_type": media_type,
            })
            content.append({
                "type": "text",
                "text": f"Screenshot {i + 1}"
            })
        
        content.append({
            "type": "text",
            "text": "\n\nAnalyze this recording and create a comprehensive markdown workflow document."
        })
        
        return content
    
    def _build_extraction_message_from_session(
        self,
        frames: list,  # List of FrameInfo
        transcript,  # Transcript or None
        duration: float,
    ) -> list[dict]:
        """Build message content from a processed session."""
        content = []
        
        # Add text description
        text_parts = ["# Video Recording Analysis\n\n"]
        text_parts.append(f"Duration: {duration:.1f} seconds\n")
        text_parts.append(f"Frames extracted: {len(frames)}\n\n")
        
        # Add transcript if available
        if transcript and transcript.text:
            text_parts.append("## Voice-Over Transcript\n\n")
            text_parts.append(f"{transcript.text}\n")
            
            if transcript.segments:
                text_parts.append("\n### Segments with timestamps:\n")
                for seg in transcript.segments:
                    text_parts.append(f"- [{seg.start:.1f}s - {seg.end:.1f}s]: {seg.text}\n")
        else:
            text_parts.append("## Voice-Over\n\nNo audio narration was detected in this recording.\n")
        
        content.append({"type": "text", "text": "".join(text_parts)})
        
        # Add frames as images
        content.append({
            "type": "text",
            "text": f"\n## Screenshots ({len(frames)} frames)\n\nBelow are frames from the recording in chronological order:\n"
        })
        
        for i, frame in enumerate(frames):
            # Resize and compress image to avoid API size limits
            image_data, media_type = self._resize_and_encode_image(frame.path)
            
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data,
                }
            })
            content.append({
                "type": "text",
                "text": f"Frame {i + 1} at {frame.timestamp:.1f}s"
            })
        
        # Add extraction instruction
        content.append({
            "type": "text",
            "text": "\n\nPlease analyze this recording and create a comprehensive markdown workflow document. Include the YAML frontmatter with parameters, then detailed instructions with reasoning."
        })
        
        return content
    
    def _build_openai_message_from_session(
        self,
        frames: list,  # List of FrameInfo
        transcript,  # Transcript or None
        duration: float,
    ) -> list[dict]:
        """Build message content in OpenAI's vision API format."""
        content = []
        
        # Add text description
        text_parts = ["# Video Recording Analysis\n\n"]
        text_parts.append(f"Duration: {duration:.1f} seconds\n")
        text_parts.append(f"Frames extracted: {len(frames)}\n\n")
        
        # Add transcript if available
        if transcript and transcript.text:
            text_parts.append("## Voice-Over Transcript\n\n")
            text_parts.append(f"{transcript.text}\n")
            
            if transcript.segments:
                text_parts.append("\n### Segments with timestamps:\n")
                for seg in transcript.segments:
                    text_parts.append(f"- [{seg.start:.1f}s - {seg.end:.1f}s]: {seg.text}\n")
        else:
            text_parts.append("## Voice-Over\n\nNo audio narration was detected in this recording.\n")
        
        content.append({"type": "text", "text": "".join(text_parts)})
        
        # Add frames as images (OpenAI format)
        content.append({
            "type": "text",
            "text": f"\n## Screenshots ({len(frames)} frames)\n\nBelow are frames from the recording in chronological order:\n"
        })
        
        for i, frame in enumerate(frames):
            # Resize and compress image to avoid API size limits
            image_data, media_type = self._resize_and_encode_image(frame.path)
            
            # OpenAI uses image_url with data URL format
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{image_data}"
                }
            })
            content.append({
                "type": "text",
                "text": f"Frame {i + 1} at {frame.timestamp:.1f}s"
            })
        
        # Add extraction instruction
        content.append({
            "type": "text",
            "text": "\n\nPlease analyze this recording and create a comprehensive markdown workflow document. Include the YAML frontmatter with parameters, then detailed instructions with reasoning."
        })
        
        return content
    
    def _build_extraction_message(
        self,
        session_data: dict,
        screenshot_paths: list[Path],
    ) -> list[dict]:
        """Build the message content for extraction (legacy format)."""
        content = []
        
        # Add text description
        text_parts = ["# Recording Session Analysis\n\n"]
        text_parts.append(f"Session ID: {session_data.get('session_id', 'unknown')}\n")
        text_parts.append(f"Duration: {session_data.get('duration', 0):.1f} seconds\n\n")
        
        # Add events summary if present
        events = session_data.get("events", [])
        if events:
            text_parts.append("## Input Events\n\n")
            for event in events[:100]:
                timestamp = event.get("timestamp", 0)
                if "start_time" in session_data:
                    timestamp = timestamp - session_data["start_time"]
                text_parts.append(f"- [{timestamp:.2f}s] {event.get('event_type', 'unknown')}: {event.get('data', {})}\n")
        
        # Add transcript if available
        if session_data.get("transcript"):
            transcript = session_data["transcript"]
            text_parts.append("\n## Voice-Over Transcript\n\n")
            text_parts.append(f"{transcript.get('text', '')}\n")
            
            if transcript.get("segments"):
                text_parts.append("\n### Segments with timestamps:\n")
                for seg in transcript["segments"]:
                    text_parts.append(f"- [{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text']}\n")
        
        content.append({"type": "text", "text": "".join(text_parts)})
        
        # Add screenshots
        if screenshot_paths:
            content.append({
                "type": "text",
                "text": f"\n## Screenshots ({len(screenshot_paths)} images)\n\nBelow are screenshots from the recording in chronological order:\n"
            })
            
            for i, path in enumerate(screenshot_paths):
                # Resize and compress image to avoid API size limits
                image_data, media_type = self._resize_and_encode_image(path)
                
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    }
                })
                content.append({
                    "type": "text",
                    "text": f"Screenshot {i + 1} of {len(screenshot_paths)}"
                })
        
        # Add extraction instruction
        content.append({
            "type": "text",
            "text": "\n\nPlease analyze this recording and create a comprehensive markdown workflow document. Include the YAML frontmatter with parameters, then detailed instructions with reasoning."
        })
        
        return content
    
    def _build_openai_extraction_message(
        self,
        session_data: dict,
        screenshot_paths: list[Path],
    ) -> list[dict]:
        """Build the message content for extraction in OpenAI format (legacy format)."""
        content = []
        
        # Add text description
        text_parts = ["# Recording Session Analysis\n\n"]
        text_parts.append(f"Session ID: {session_data.get('session_id', 'unknown')}\n")
        text_parts.append(f"Duration: {session_data.get('duration', 0):.1f} seconds\n\n")
        
        # Add events summary if present
        events = session_data.get("events", [])
        if events:
            text_parts.append("## Input Events\n\n")
            for event in events[:100]:
                timestamp = event.get("timestamp", 0)
                if "start_time" in session_data:
                    timestamp = timestamp - session_data["start_time"]
                text_parts.append(f"- [{timestamp:.2f}s] {event.get('event_type', 'unknown')}: {event.get('data', {})}\n")
        
        # Add transcript if available
        if session_data.get("transcript"):
            transcript = session_data["transcript"]
            text_parts.append("\n## Voice-Over Transcript\n\n")
            text_parts.append(f"{transcript.get('text', '')}\n")
            
            if transcript.get("segments"):
                text_parts.append("\n### Segments with timestamps:\n")
                for seg in transcript["segments"]:
                    text_parts.append(f"- [{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text']}\n")
        
        content.append({"type": "text", "text": "".join(text_parts)})
        
        # Add screenshots (OpenAI format)
        if screenshot_paths:
            content.append({
                "type": "text",
                "text": f"\n## Screenshots ({len(screenshot_paths)} images)\n\nBelow are screenshots from the recording in chronological order:\n"
            })
            
            for i, path in enumerate(screenshot_paths):
                # Resize and compress image to avoid API size limits
                image_data, media_type = self._resize_and_encode_image(path)
                
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_data}"
                    }
                })
                content.append({
                    "type": "text",
                    "text": f"Screenshot {i + 1} of {len(screenshot_paths)}"
                })
        
        # Add extraction instruction
        content.append({
            "type": "text",
            "text": "\n\nPlease analyze this recording and create a comprehensive markdown workflow document. Include the YAML frontmatter with parameters, then detailed instructions with reasoning."
        })
        
        return content
    
    def extract_from_data(
        self,
        screenshots: list[tuple[Path, dict]],
        events: list[dict],
        transcript_text: str | None = None,
        transcript_segments: list[dict] | None = None,
    ) -> Workflow:
        """Extract a workflow from raw data (legacy method).
        
        Args:
            screenshots: List of (screenshot_path, metadata) tuples.
            events: List of input events.
            transcript_text: Full transcript text.
            transcript_segments: Transcript segments with timestamps.
            
        Returns:
            Extracted Workflow object.
        """
        # Build unified content format
        content = self._build_data_extraction_content(
            screenshots=screenshots,
            events=events,
            transcript_text=transcript_text,
            transcript_segments=transcript_segments,
        )
        
        # Use LLMClient with synthesis model
        response_text = self.llm.generate(
            model=self.model_config.synthesis,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            content=content,
            max_tokens=8192,
            phase="data_extraction",
        )
        
        markdown_content = self._extract_markdown(response_text)
        return Workflow.from_markdown(markdown_content)
    
    def _build_data_extraction_content(
        self,
        screenshots: list[tuple[Path, dict]],
        events: list[dict],
        transcript_text: str | None,
        transcript_segments: list[dict] | None,
    ) -> list[dict]:
        """Build unified content for data extraction."""
        content = []
        
        text_parts = ["# Recording Analysis\n\n"]
        
        # Events
        if events:
            text_parts.append("## Input Events\n\n")
            for event in events[:100]:
                text_parts.append(f"- [{event.get('timestamp', 0):.2f}s] {event.get('event_type', 'unknown')}: {event.get('data', {})}\n")
        
        # Transcript
        if transcript_text:
            text_parts.append("\n## Voice-Over Transcript\n\n")
            text_parts.append(f"{transcript_text}\n")
            
            if transcript_segments:
                text_parts.append("\n### Segments:\n")
                for seg in transcript_segments:
                    text_parts.append(f"- [{seg.get('start', 0):.1f}s - {seg.get('end', 0):.1f}s]: {seg.get('text', '')}\n")
        
        content.append({"type": "text", "text": "".join(text_parts)})
        
        # Add screenshots in unified format
        if screenshots:
            content.append({
                "type": "text",
                "text": f"\n## Screenshots ({len(screenshots)} images)\n\n"
            })
            
            for i, (screenshot_path, metadata) in enumerate(screenshots):
                image_data, media_type = self._resize_and_encode_image(screenshot_path)
                
                content.append({
                    "type": "image",
                    "data": image_data,
                    "media_type": media_type,
                })
                
                timestamp = metadata.get("timestamp", i)
                content.append({
                    "type": "text",
                    "text": f"Screenshot {i + 1} at {timestamp}s"
                })
        
        content.append({
            "type": "text",
            "text": "\n\nAnalyze this recording and create a comprehensive markdown workflow document."
        })
        
        return content
    
    def _build_extraction_message_from_data(
        self,
        screenshots: list[tuple[Path, dict]],
        events: list[dict],
        transcript_text: str | None,
        transcript_segments: list[dict] | None,
    ) -> list[dict]:
        """Build message content from raw data."""
        content = []
        
        text_parts = ["# Recording Analysis\n\n"]
        
        # Events
        if events:
            text_parts.append("## Input Events\n\n")
            for event in events[:100]:
                text_parts.append(f"- [{event.get('timestamp', 0):.2f}s] {event.get('event_type', 'unknown')}: {event.get('data', {})}\n")
        
        # Transcript
        if transcript_text:
            text_parts.append("\n## Voice-Over Transcript\n\n")
            text_parts.append(f"{transcript_text}\n")
            
            if transcript_segments:
                text_parts.append("\n### Segments:\n")
                for seg in transcript_segments:
                    text_parts.append(f"- [{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text']}\n")
        
        content.append({"type": "text", "text": "".join(text_parts)})
        
        # Screenshots
        if screenshots:
            content.append({
                "type": "text",
                "text": f"\n## Screenshots ({len(screenshots)} images)\n\n"
            })
            
            for i, (path, metadata) in enumerate(screenshots):
                # Resize and compress image to avoid API size limits
                image_data, media_type = self._resize_and_encode_image(path)
                
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    }
                })
                content.append({
                    "type": "text",
                    "text": f"Screenshot {i + 1}: trigger={metadata.get('trigger', 'unknown')}"
                })
        
        content.append({
            "type": "text",
            "text": "\n\nCreate a comprehensive markdown workflow document with YAML frontmatter and detailed instructions."
        })
        
        return content
    
    def _build_openai_message_from_data(
        self,
        screenshots: list[tuple[Path, dict]],
        events: list[dict],
        transcript_text: str | None,
        transcript_segments: list[dict] | None,
    ) -> list[dict]:
        """Build message content from raw data in OpenAI format."""
        content = []
        
        text_parts = ["# Recording Analysis\n\n"]
        
        # Events
        if events:
            text_parts.append("## Input Events\n\n")
            for event in events[:100]:
                text_parts.append(f"- [{event.get('timestamp', 0):.2f}s] {event.get('event_type', 'unknown')}: {event.get('data', {})}\n")
        
        # Transcript
        if transcript_text:
            text_parts.append("\n## Voice-Over Transcript\n\n")
            text_parts.append(f"{transcript_text}\n")
            
            if transcript_segments:
                text_parts.append("\n### Segments:\n")
                for seg in transcript_segments:
                    text_parts.append(f"- [{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text']}\n")
        
        content.append({"type": "text", "text": "".join(text_parts)})
        
        # Screenshots (OpenAI format)
        if screenshots:
            content.append({
                "type": "text",
                "text": f"\n## Screenshots ({len(screenshots)} images)\n\n"
            })
            
            for i, (path, metadata) in enumerate(screenshots):
                # Resize and compress image to avoid API size limits
                image_data, media_type = self._resize_and_encode_image(path)
                
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_data}"
                    }
                })
                content.append({
                    "type": "text",
                    "text": f"Screenshot {i + 1}: trigger={metadata.get('trigger', 'unknown')}"
                })
        
        content.append({
            "type": "text",
            "text": "\n\nCreate a comprehensive markdown workflow document with YAML frontmatter and detailed instructions."
        })
        
        return content
    
    def _extract_markdown(self, response_text: str) -> str:
        """Extract markdown content from Claude's response."""
        text = response_text.strip()
        
        # If response is wrapped in code block, extract it
        if text.startswith("```markdown"):
            start = text.find("```markdown") + 11
            end = text.rfind("```")
            if end > start:
                text = text[start:end].strip()
        elif text.startswith("```"):
            start = text.find("```") + 3
            # Skip language identifier if present
            newline = text.find("\n", start)
            if newline != -1:
                start = newline + 1
            end = text.rfind("```")
            if end > start:
                text = text[start:end].strip()
        
        # Ensure we have frontmatter
        if not text.startswith("---"):
            # Try to add minimal frontmatter
            text = f"---\nid: extracted_workflow\nname: Extracted Workflow\n---\n\n{text}"
        
        return text
    
    # =========================================================================
    # Multi-Pass Extraction Methods
    # =========================================================================
    
    def _detect_events_pass(
        self,
        frames: list,  # List of FrameInfo
        transcript,  # Transcript or None
        chunk_size: int = 10,
        overlap: int = 2,
        verbose: bool = True,
        parallel: bool = True,
    ) -> list[DetectedEvent]:
        """Pass 1: Detect events from frames in overlapping chunks.
        
        Processes all frames in chunks, with overlap to catch actions that
        span chunk boundaries. Each chunk is analyzed independently to
        identify discrete user actions.
        
        Args:
            frames: List of FrameInfo objects from the video.
            transcript: Optional Transcript with voice-over text.
            chunk_size: Number of frames per chunk.
            overlap: Number of frames to overlap between chunks.
            verbose: Whether to print progress information.
            parallel: Whether to use parallel batch processing (faster for Gemini).
            
        Returns:
            List of DetectedEvent objects in chronological order.
        """
        from utils.tracking import detect_provider
        
        all_events: list[DetectedEvent] = []
        total_frames = len(frames)
        
        if total_frames == 0:
            return all_events
        
        # Create chunks with overlap
        chunks = self._create_overlapping_chunks(frames, chunk_size, overlap)
        total_chunks = len(chunks)
        
        if verbose:
            self._log("PASS 1: Event Detection", level="header")
            self._log(f"Total frames: {total_frames}")
            self._log(f"Chunk size: {chunk_size} frames (overlap: {overlap})")
            self._log(f"Total chunks to process: {total_chunks}")
        
        # Check if we should use parallel processing (only for Gemini)
        provider = detect_provider(self.model_config.event_detection)
        use_parallel = parallel and provider == "gemini"
        
        if use_parallel:
            # Parallel batch processing for Gemini
            all_events = self._detect_events_parallel(
                chunks=chunks,
                transcript=transcript,
                verbose=verbose,
            )
        else:
            # Sequential processing (for Anthropic/OpenAI or when parallel=False)
            all_events = self._detect_events_sequential(
                chunks=chunks,
                transcript=transcript,
                overlap=overlap,
                verbose=verbose,
            )
        
        if verbose:
            self._log(f"Pass 1 complete: Detected {len(all_events)} events", level="success")
        
        return all_events
    
    def _detect_events_sequential(
        self,
        chunks: list,
        transcript,
        overlap: int,
        verbose: bool,
    ) -> list[DetectedEvent]:
        """Sequential event detection (original approach)."""
        all_events: list[DetectedEvent] = []
        
        # Use tqdm for progress bar
        chunk_iterator = tqdm(
            enumerate(chunks),
            total=len(chunks),
            desc="Detecting events",
            unit="chunk",
            disable=not verbose,
        )
        
        for chunk_idx, (chunk_frames, start_idx) in chunk_iterator:
            # Update progress bar description with frame range
            chunk_iterator.set_postfix_str(
                f"frames={start_idx + 1}-{start_idx + len(chunk_frames)}, events={len(all_events)}"
            )
            
            # Get relevant transcript segments for this chunk
            chunk_start_time = chunk_frames[0].timestamp
            chunk_end_time = chunk_frames[-1].timestamp
            relevant_transcript = self._get_transcript_for_timerange(
                transcript, chunk_start_time, chunk_end_time
            )
            
            # Detect events in this chunk (passes tqdm bar for token updates)
            chunk_events = self._detect_events_in_chunk(
                frames=chunk_frames,
                start_frame_idx=start_idx,
                transcript_text=relevant_transcript,
                tqdm_bar=chunk_iterator,
            )
            
            # Merge events, handling overlap with previous chunk
            all_events = self._merge_events(all_events, chunk_events, overlap > 0)
        
        return all_events
    
    def _detect_events_parallel(
        self,
        chunks: list,
        transcript,
        verbose: bool,
    ) -> list[DetectedEvent]:
        """Parallel batch event detection for Gemini (5 requests at a time)."""
        from prompts.analyzer_prompts import EVENT_DETECTION_PROMPT
        
        total_chunks = len(chunks)
        
        if verbose:
            self._log(f"Using parallel batch processing (5 chunks at a time)")
        
        # Build all requests upfront
        requests = []
        for chunk_idx, (chunk_frames, start_idx) in enumerate(chunks):
            # Get relevant transcript segments for this chunk
            chunk_start_time = chunk_frames[0].timestamp
            chunk_end_time = chunk_frames[-1].timestamp
            relevant_transcript = self._get_transcript_for_timerange(
                transcript, chunk_start_time, chunk_end_time
            )
            
            # Build message content
            content = self._build_event_detection_message(
                frames=chunk_frames,
                start_frame_idx=start_idx,
                transcript_text=relevant_transcript,
            )
            
            requests.append({
                "model": self.model_config.event_detection,
                "system_prompt": EVENT_DETECTION_PROMPT,
                "content": content,
                "max_tokens": 4096,
                "parse_json": True,
                "json_type": "array",
                "default_json": [],
                "phase": "pass1_events",
            })
        
        # Progress bar for parallel processing
        pbar = tqdm(
            total=total_chunks,
            desc="Detecting events (parallel)",
            unit="chunk",
            disable=not verbose,
        )
        
        def progress_callback(completed: int, total: int) -> None:
            pbar.n = completed
            pbar.refresh()
        
        try:
            # Fire requests in parallel batches
            results = self.llm.generate_batch_parallel(
                requests=requests,
                batch_size=5,  # Gemini free tier limit
                batch_wait=60.0,  # Wait for rate limit reset
                progress_callback=progress_callback,
            )
        finally:
            pbar.close()
        
        # Parse results and merge events
        all_events: list[DetectedEvent] = []
        for chunk_idx, events_data in enumerate(results):
            chunk_events = self._parse_events_from_data(events_data)
            # Merge events, handling overlap
            all_events = self._merge_events(all_events, chunk_events, True)
        
        return all_events
    
    def _create_overlapping_chunks(
        self,
        frames: list,
        chunk_size: int,
        overlap: int,
    ) -> list[tuple[list, int]]:
        """Create overlapping chunks of frames.
        
        Returns list of (chunk_frames, start_index) tuples.
        """
        chunks = []
        step = max(1, chunk_size - overlap)
        
        for i in range(0, len(frames), step):
            chunk = frames[i:i + chunk_size]
            if len(chunk) > 1:  # Need at least 2 frames to detect changes
                chunks.append((chunk, i))
        
        return chunks
    
    def _get_transcript_for_timerange(
        self,
        transcript,
        start_time: float,
        end_time: float,
    ) -> str:
        """Extract transcript text relevant to a time range."""
        if not transcript or not transcript.segments:
            return ""
        
        relevant_parts = []
        for segment in transcript.segments:
            # Check if segment overlaps with our time range
            if segment.end >= start_time and segment.start <= end_time:
                relevant_parts.append(f"[{segment.start:.1f}s]: {segment.text}")
        
        return "\n".join(relevant_parts)
    
    def _detect_events_in_chunk(
        self,
        frames: list,
        start_frame_idx: int,
        transcript_text: str,
        *,
        tqdm_bar: Any = None,
    ) -> list[DetectedEvent]:
        """Detect events in a single chunk of frames.
        
        Args:
            frames: List of frames to analyze.
            start_frame_idx: Starting frame index in the full video.
            transcript_text: Relevant transcript text for this chunk.
            tqdm_bar: Optional tqdm progress bar to update with cumulative tokens.
        """
        # Build message content in unified format
        content = self._build_event_detection_message(
            frames=frames,
            start_frame_idx=start_frame_idx,
            transcript_text=transcript_text,
        )
        
        # Use LLMClient with the event detection model
        events_data = self.llm.generate(
            model=self.model_config.event_detection,
            system_prompt=EVENT_DETECTION_PROMPT,
            content=content,
            max_tokens=4096,
            parse_json=True,
            json_type="array",
            default_json=[],
            tqdm_bar=tqdm_bar,
            phase="pass1_events",
        )
        
        # Parse events from JSON data
        return self._parse_events_from_data(events_data)
    
    def _build_event_detection_message(
        self,
        frames: list,
        start_frame_idx: int,
        transcript_text: str,
    ) -> list[dict]:
        """Build message content for event detection in unified format."""
        content = []
        
        # Add context
        text_parts = [f"# Frame Chunk Analysis\n\n"]
        text_parts.append(f"Analyzing frames {start_frame_idx + 1} to {start_frame_idx + len(frames)}\n")
        text_parts.append(f"Time range: {frames[0].timestamp:.1f}s to {frames[-1].timestamp:.1f}s\n\n")
        
        if transcript_text:
            text_parts.append("## Relevant Voice-Over:\n")
            text_parts.append(f"{transcript_text}\n\n")
        
        content.append({"type": "text", "text": "".join(text_parts)})
        
        # Add frames header
        content.append({
            "type": "text",
            "text": f"## Frames ({len(frames)} images)\n\n"
        })
        
        for i, frame in enumerate(frames):
            # Resize and compress image, then add in unified format
            image_data, media_type = self._resize_and_encode_image(frame.path)
            
            # Unified format: image with base64 data
            content.append({
                "type": "image",
                "data": image_data,
                "media_type": media_type,
            })
            
            content.append({
                "type": "text",
                "text": f"Frame {start_frame_idx + i + 1} at {frame.timestamp:.1f}s"
            })
        
        content.append({
            "type": "text",
            "text": "\n\nAnalyze the changes between these frames and detect all user actions. Output as JSON array."
        })
        
        return content
    
    def _parse_events_from_data(self, events_data: list) -> list[DetectedEvent]:
        """Parse events from already-parsed JSON data."""
        events = []
        for event_dict in events_data:
            try:
                event = DetectedEvent.from_dict(event_dict)
                events.append(event)
            except (KeyError, ValueError) as e:
                self._log(f"Warning: Failed to parse event: {e}", level="warning")
        return events
    
    def _parse_events_response(self, response_text: str) -> list[DetectedEvent]:
        """Parse JSON events from model response."""
        events_data = extract_json_from_response(response_text, json_type="array", default=[])
        return [DetectedEvent.from_dict(event_dict) for event_dict in events_data]
    
    def _merge_events(
        self,
        existing_events: list[DetectedEvent],
        new_events: list[DetectedEvent],
        has_overlap: bool,
    ) -> list[DetectedEvent]:
        """Merge new events with existing events, handling overlap."""
        if not existing_events:
            return new_events
        
        if not new_events:
            return existing_events
        
        if not has_overlap:
            return existing_events + new_events
        
        # Check for duplicate events in the overlap region
        # Use timestamp and action_type to detect duplicates
        last_existing_time = existing_events[-1].timestamp
        
        merged = list(existing_events)
        for event in new_events:
            # Skip if this looks like a duplicate from overlap
            is_duplicate = False
            if event.timestamp <= last_existing_time:
                for existing in existing_events[-3:]:  # Check last few events
                    if (abs(event.timestamp - existing.timestamp) < 0.5 and
                        event.action_type == existing.action_type and
                        event.target == existing.target):
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                merged.append(event)
        
        return merged
    
    def _build_understanding_pass(
        self,
        events: list[DetectedEvent],
        transcript,  # Transcript or None
        batch_size: int = 15,
        verbose: bool = True,
    ) -> RunningUnderstanding:
        """Pass 2: Build running understanding from detected events.
        
        Processes events in batches, incrementally building and refining
        the workflow understanding. Each batch updates the accumulated state.
        
        Args:
            events: List of DetectedEvent from Pass 1.
            transcript: Optional Transcript for additional context.
            batch_size: Number of events per batch.
            verbose: Whether to print progress information.
            
        Returns:
            Complete RunningUnderstanding of the workflow.
        """
        understanding = RunningUnderstanding.empty()
        
        if not events:
            return understanding
        
        # Process events in batches
        total_events = len(events)
        num_batches = (total_events + batch_size - 1) // batch_size
        
        if verbose:
            self._log("PASS 2: Building Understanding", level="header")
            self._log(f"Total events to process: {total_events}")
            self._log(f"Batch size: {batch_size} events")
            self._log(f"Total batches: {num_batches}")
        
        # Use tqdm for progress bar
        batch_iterator = tqdm(
            range(num_batches),
            desc="Building understanding",
            unit="batch",
            disable=not verbose,
        )
        
        for batch_idx in batch_iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_events)
            batch_events = events[start_idx:end_idx]
            
            # Update progress bar description
            batch_iterator.set_postfix_str(
                f"events={start_idx + 1}-{end_idx}, steps={len(understanding.steps)}"
            )
            
            # Get transcript context for this batch's time range
            if batch_events:
                batch_start_time = batch_events[0].timestamp
                batch_end_time = batch_events[-1].timestamp
                transcript_context = self._get_transcript_for_timerange(
                    transcript, batch_start_time, batch_end_time
                )
            else:
                transcript_context = ""
            
            # Update understanding with this batch (passes tqdm bar for token updates)
            understanding = self._update_understanding(
                current_understanding=understanding,
                new_events=batch_events,
                transcript_context=transcript_context,
                is_first_batch=(batch_idx == 0),
                is_last_batch=(batch_idx == num_batches - 1),
                tqdm_bar=batch_iterator,
            )
        
        if verbose:
            self._log(f"Pass 2 complete: Built understanding with {len(understanding.steps)} steps", level="success")
            if understanding.task_goal:
                self._log(f"  Task goal: {understanding.task_goal[:80]}...")
            self._log(f"  Parameters detected: {len(understanding.parameters)}")
        
        return understanding
    
    def _update_understanding(
        self,
        current_understanding: RunningUnderstanding,
        new_events: list[DetectedEvent],
        transcript_context: str,
        is_first_batch: bool,
        is_last_batch: bool,
        *,
        tqdm_bar: Any = None,
    ) -> RunningUnderstanding:
        """Update understanding with a new batch of events.
        
        Args:
            current_understanding: Current accumulated understanding.
            new_events: New events to incorporate.
            transcript_context: Relevant transcript text.
            is_first_batch: Whether this is the first batch.
            is_last_batch: Whether this is the last batch.
            tqdm_bar: Optional tqdm progress bar to update with cumulative tokens.
        """
        # Build message content
        content = self._build_understanding_message(
            current_understanding=current_understanding,
            new_events=new_events,
            transcript_context=transcript_context,
            is_first_batch=is_first_batch,
            is_last_batch=is_last_batch,
        )
        
        # Use LLMClient with the understanding model
        response_text = self.llm.generate(
            model=self.model_config.understanding,
            system_prompt=UNDERSTANDING_UPDATE_PROMPT,
            content=content,
            max_tokens=16384,
            tqdm_bar=tqdm_bar,
            phase="pass2_understanding",
        )
        
        # Parse the updated understanding
        return self._parse_understanding_response(
            response_text, current_understanding
        )
    
    def _build_understanding_message(
        self,
        current_understanding: RunningUnderstanding,
        new_events: list[DetectedEvent],
        transcript_context: str,
        is_first_batch: bool,
        is_last_batch: bool,
    ) -> list[dict]:
        """Build message content for understanding update."""
        parts = []
        
        # Current state
        if is_first_batch:
            parts.append("# Initial Workflow Analysis\n\n")
            parts.append("This is the first batch of events. Start building the workflow understanding from scratch.\n\n")
        else:
            parts.append("# Workflow Understanding Update\n\n")
            parts.append("## Current Understanding\n\n")
            parts.append(f"```json\n{json.dumps(current_understanding.to_dict(), indent=2)}\n```\n\n")
        
        if is_last_batch:
            parts.append("**Note:** This is the final batch of events. Finalize the understanding.\n\n")
        
        # New events
        parts.append("## New Events to Process\n\n")
        events_data = [e.to_dict() for e in new_events]
        parts.append(f"```json\n{json.dumps(events_data, indent=2)}\n```\n\n")
        
        # Transcript context
        if transcript_context:
            parts.append("## Voice-Over Context\n\n")
            parts.append(f"{transcript_context}\n\n")
        
        parts.append("Update the understanding based on these new events. Output the complete updated understanding as JSON.")
        
        return [{"type": "text", "text": "".join(parts)}]
    
    def _parse_understanding_response(
        self,
        response_text: str,
        fallback: RunningUnderstanding,
    ) -> RunningUnderstanding:
        """Parse understanding JSON from model response."""
        data = extract_json_from_response(response_text, json_type="object", default=None)
        if data is not None:
            try:
                return RunningUnderstanding.from_dict(data)
            except KeyError as e:
                self._log(f"Failed to parse understanding response: {e}", level="warning")
        else:
            self._log(f"No valid JSON found in response (length: {len(response_text)} chars), falling back to previous understanding", level="warning")
        return fallback
    
    def _generate_workflow_pass(
        self,
        understanding: RunningUnderstanding,
        verbose: bool = True,
    ) -> Workflow:
        """Pass 3: Generate final workflow from understanding.
        
        Takes the complete accumulated understanding and synthesizes
        a polished markdown workflow document.
        
        Args:
            understanding: Complete RunningUnderstanding from Pass 2.
            verbose: Whether to print progress information.
            
        Returns:
            Final Workflow object.
        """
        if verbose:
            self._log("PASS 3: Workflow Synthesis", level="header")
            self._log(f"Synthesizing workflow from {len(understanding.steps)} steps...")
        
        # Build message content
        content = self._build_synthesis_message(understanding)
        
        # Use tqdm with a simple indeterminate progress (single iteration)
        with tqdm(total=1, desc="Generating workflow", unit="doc", disable=not verbose) as pbar:
            # Use LLMClient with the synthesis model
            response_text = self.llm.generate(
                model=self.model_config.synthesis,
                system_prompt=WORKFLOW_SYNTHESIS_PROMPT,
                content=content,
                max_tokens=8192,
                tqdm_bar=pbar,
                phase="pass3_synthesis",
            )
            pbar.update(1)
        
        # Parse the markdown response
        markdown_content = self._extract_markdown(response_text)
        workflow = Workflow.from_markdown(markdown_content)
        
        if verbose:
            self._log(f"Pass 3 complete!", level="success")
            self._log(f"  Workflow: {workflow.name}")
            self._log(f"  Parameters: {len(workflow.parameters)}")
        
        return workflow
    
    def _build_synthesis_message(
        self,
        understanding: RunningUnderstanding,
    ) -> list[dict]:
        """Build message content for workflow synthesis."""
        parts = []
        
        parts.append("# Complete Workflow Understanding\n\n")
        parts.append("Based on analyzing a screen recording, here is the complete understanding of the workflow:\n\n")
        parts.append(f"```json\n{json.dumps(understanding.to_dict(), indent=2)}\n```\n\n")
        parts.append("Please synthesize this into a comprehensive, polished markdown workflow document following the specified format.")
        
        return [{"type": "text", "text": "".join(parts)}]
