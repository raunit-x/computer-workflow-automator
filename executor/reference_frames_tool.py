"""Reference frames tool for retrieving original recording frames during execution."""

import base64
import json
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from .macos_computer import ToolResult, ToolError

# Image constraints for API
MAX_LONG_EDGE = 1568
MAX_PIXELS = 1_150_000
JPEG_QUALITY = 80


@dataclass
class ReferenceFramesTool:
    """Tool for retrieving reference frames from the original recording.
    
    When the agent gets stuck on a step, it can request the reference frames
    to see what the screen looked like when the human performed that step
    in the original recording.
    """
    
    name: str = "reference_frames"
    processed_session_path: str | None = None
    reference_frames_metadata: dict | None = None
    
    def __post_init__(self):
        """Load reference frames metadata if available."""
        if self.processed_session_path and not self.reference_frames_metadata:
            self._load_metadata_from_session()
    
    def _load_metadata_from_session(self) -> None:
        """Load processed session metadata to get frame paths."""
        session_path = Path(self.processed_session_path)
        metadata_file = session_path / "processed_session.json"
        
        if metadata_file.exists():
            with open(metadata_file) as f:
                data = json.load(f)
                # Extract frame info for quick lookup
                self.reference_frames_metadata = {
                    "frames": data.get("frames", []),
                    "duration": data.get("duration", 0),
                }
    
    def extract_reference_metadata_from_instructions(
        self,
        instructions: str,
    ) -> dict | None:
        """Extract reference frames metadata from workflow instructions.
        
        The metadata is stored as a hidden HTML comment in the instructions.
        """
        pattern = r'<!-- REFERENCE_FRAMES_METADATA\n(.*?)\n-->'
        match = re.search(pattern, instructions, re.DOTALL)
        
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return None
        return None
    
    def set_workflow_context(
        self,
        processed_session_path: str | None,
        instructions: str,
    ) -> None:
        """Set the workflow context for reference frame lookups.
        
        Args:
            processed_session_path: Path to the processed session folder.
            instructions: Workflow instructions (may contain embedded metadata).
        """
        self.processed_session_path = processed_session_path
        
        # Try to load metadata from session
        if processed_session_path:
            self._load_metadata_from_session()
        
        # Also extract step-specific frame mappings from instructions
        embedded_metadata = self.extract_reference_metadata_from_instructions(instructions)
        if embedded_metadata:
            if self.reference_frames_metadata is None:
                self.reference_frames_metadata = {}
            self.reference_frames_metadata["step_reference_frames"] = embedded_metadata.get(
                "step_reference_frames", {}
            )
    
    def _resize_and_encode_image(self, image_path: Path) -> str:
        """Resize image if needed and return base64-encoded JPEG."""
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Check if resizing is needed
            width, height = img.size
            max_dim = max(width, height)
            total_pixels = width * height
            
            # Calculate scale factor
            scale = 1.0
            if max_dim > MAX_LONG_EDGE:
                scale = min(scale, MAX_LONG_EDGE / max_dim)
            if total_pixels > MAX_PIXELS:
                scale = min(scale, (MAX_PIXELS / total_pixels) ** 0.5)
            
            if scale < 1.0:
                new_width = int(width * scale)
                new_height = int(height * scale)
                try:
                    resample = Image.Resampling.LANCZOS
                except AttributeError:
                    resample = Image.LANCZOS  # type: ignore
                img = img.resize((new_width, new_height), resample)
            
            # Encode as JPEG
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=JPEG_QUALITY, optimize=True)
            buffer.seek(0)
            
            return base64.b64encode(buffer.read()).decode()
    
    async def __call__(
        self,
        *,
        step_number: int | None = None,
        timestamp: float | None = None,
        count: int = 3,
        **kwargs,
    ) -> ToolResult:
        """Get reference frames for a step or timestamp.
        
        Args:
            step_number: The workflow step number to get frames for.
            timestamp: Specific timestamp in the recording to get frames near.
            count: Maximum number of frames to return (default 3).
            
        Returns:
            ToolResult with reference frame images and descriptions.
        """
        if not self.processed_session_path:
            return ToolResult(
                error="No processed session path configured. Reference frames unavailable."
            )
        
        if not self.reference_frames_metadata:
            return ToolResult(
                error="No reference frames metadata available."
            )
        
        frame_paths: list[str] = []
        
        # If step_number is provided, try to get step-specific frames
        if step_number is not None:
            step_frames = self.reference_frames_metadata.get("step_reference_frames", {})
            step_data = step_frames.get(str(step_number), {})
            frame_paths = step_data.get("paths", [])[:count]
        
        # If timestamp is provided or no step frames found, find frames near timestamp
        if not frame_paths and timestamp is not None:
            all_frames = self.reference_frames_metadata.get("frames", [])
            # Sort by distance to target timestamp
            sorted_frames = sorted(
                all_frames,
                key=lambda f: abs(f.get("timestamp", 0) - timestamp)
            )
            frame_paths = [f.get("path") for f in sorted_frames[:count]]
        
        # If still no frames, get representative frames from the session
        if not frame_paths:
            all_frames = self.reference_frames_metadata.get("frames", [])
            if all_frames:
                # Get frames evenly distributed across the recording
                step_size = max(1, len(all_frames) // count)
                frame_paths = [
                    all_frames[i].get("path")
                    for i in range(0, len(all_frames), step_size)
                ][:count]
        
        if not frame_paths:
            return ToolResult(
                error="No reference frames found for the specified criteria."
            )
        
        # Load and encode the frames
        encoded_frames: list[tuple[str, str]] = []  # (base64_data, description)
        
        for frame_path in frame_paths:
            path = Path(frame_path)
            if not path.exists():
                # Try relative to processed session path
                path = Path(self.processed_session_path) / "frames" / path.name
            
            if path.exists():
                try:
                    base64_data = self._resize_and_encode_image(path)
                    # Extract timestamp from filename or metadata
                    description = f"Reference frame: {path.name}"
                    encoded_frames.append((base64_data, description))
                except Exception as e:
                    continue  # Skip frames that can't be loaded
        
        if not encoded_frames:
            return ToolResult(
                error="Failed to load any reference frames."
            )
        
        # Return the first frame as the main image, with descriptions of all
        descriptions = [desc for _, desc in encoded_frames]
        output_text = (
            f"Retrieved {len(encoded_frames)} reference frame(s) from the original recording.\n"
            f"These show what the screen looked like when the human performed this step.\n"
            f"Frames: {', '.join(descriptions)}"
        )
        
        # Return the first frame image (API limitation: one image per result)
        return ToolResult(
            output=output_text,
            base64_image=encoded_frames[0][0] if encoded_frames else None,
        )
    
    def to_params(self) -> dict:
        """Return tool parameters for Claude API."""
        return {
            "name": self.name,
            "description": (
                "Retrieve reference frames from the original screen recording. "
                "Use this when you're stuck on a step and need to see what the screen "
                "looked like when the human performed this action. Provide either a "
                "step_number to get frames for a specific workflow step, or a timestamp "
                "to get frames near a specific point in the recording."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "step_number": {
                        "type": "integer",
                        "description": "The workflow step number to get reference frames for.",
                    },
                    "timestamp": {
                        "type": "number",
                        "description": "Timestamp in seconds to get frames near.",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Maximum number of frames to return (default 3).",
                        "default": 3,
                    },
                },
            },
        }

