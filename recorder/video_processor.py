"""Video processor for extracting frames and audio from video files."""

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .transcriber import Transcriber, Transcript

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class FrameInfo:
    """Information about an extracted frame."""
    
    path: Path
    timestamp: float  # Timestamp in seconds
    index: int
    
    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "path": str(self.path),
            "timestamp": self.timestamp,
            "index": self.index,
        }


@dataclass
class ProcessedSession:
    """Result of processing a video file."""
    
    session_id: str
    video_path: Path
    output_dir: Path
    frames: list[FrameInfo]
    transcript: Transcript | None
    audio_path: Path | None
    duration: float
    fps: float = 2.0  # Frames per second extracted
    
    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "session_id": self.session_id,
            "video_path": str(self.video_path),
            "output_dir": str(self.output_dir),
            "frames": [f.to_dict() for f in self.frames],
            "transcript": self.transcript.to_dict() if self.transcript else None,
            "audio_path": str(self.audio_path) if self.audio_path else None,
            "duration": self.duration,
            "fps": self.fps,
        }
    
    def save_metadata(self) -> Path:
        """Save session metadata to JSON file."""
        metadata_path = self.output_dir / "processed_session.json"
        print(f"Saving session metadata to {metadata_path}")
        with open(metadata_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Session metadata saved successfully ({len(self.frames)} frames, duration: {self.duration:.2f}s)")
        return metadata_path
    
    @classmethod
    def load(cls, output_dir: Path) -> "ProcessedSession":
        """Load a processed session from a directory."""
        metadata_path = output_dir / "processed_session.json"
        print(f"Loading processed session from {metadata_path}")
        with open(metadata_path) as f:
            data = json.load(f)
        
        frames = [
            FrameInfo(
                path=Path(f["path"]),
                timestamp=f["timestamp"],
                index=f["index"],
            )
            for f in data.get("frames", [])
        ]
        
        transcript = None
        if data.get("transcript"):
            transcript = Transcript.from_dict(data["transcript"])
        
        session = cls(
            session_id=data["session_id"],
            video_path=Path(data["video_path"]),
            output_dir=Path(data["output_dir"]),
            frames=frames,
            transcript=transcript,
            audio_path=Path(data["audio_path"]) if data.get("audio_path") else None,
            duration=data.get("duration", 0.0),
            fps=data.get("fps", 2.0),
        )
        print(f"Loaded session {session.session_id}: {len(frames)} frames, {session.duration:.2f}s duration")
        return session
    
    def get_frame_paths(self) -> list[Path]:
        """Get paths to all frame images."""
        return [f.path for f in self.frames]


class VideoProcessor:
    """Processes video files to extract frames and audio for analysis."""
    
    def __init__(
        self,
        fps: float = 2.0,
        openai_api_key: str | None = None,
    ):
        """Initialize the video processor.
        
        Args:
            fps: Frames per second to extract (default: 2 = 1 frame every 500ms).
            openai_api_key: OpenAI API key for transcription.
        """
        self.fps = fps
        print(f"Initializing VideoProcessor with fps={fps}")
        self.transcriber = Transcriber(api_key=openai_api_key)
        
        # Verify ffmpeg is available
        print("Checking for ffmpeg availability...")
        if not self._check_ffmpeg():
            logger.error("ffmpeg not found on system PATH")
            raise RuntimeError(
                "ffmpeg not found. Please install it:\n"
                "  macOS: brew install ffmpeg\n"
                "  Ubuntu: apt-get install ffmpeg\n"
            )
        print("ffmpeg found successfully")
    
    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available."""
        return shutil.which("ffmpeg") is not None
    
    def process(
        self,
        video_path: Path,
        output_dir: Path | None = None,
        extract_audio: bool = True,
        transcribe: bool = True,
    ) -> ProcessedSession:
        """Process a video file to extract frames and optionally audio.
        
        Args:
            video_path: Path to the video file (.mov, .mp4, etc.).
            output_dir: Directory to store extracted frames and audio.
                       If None, creates a temp directory.
            extract_audio: Whether to extract the audio track.
            transcribe: Whether to transcribe the audio (requires extract_audio=True).
            
        Returns:
            ProcessedSession with frames and transcript.
        """
        video_path = Path(video_path)
        print(f"Starting video processing: {video_path}")
        
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Create output directory
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="video_processed_"))
            print(f"Created temporary output directory: {output_dir}")
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Using output directory: {output_dir}")
        
        # Get video duration
        print("Retrieving video duration...")
        duration = self._get_video_duration(video_path)
        print(f"Video duration: {duration:.2f} seconds")
        
        # Generate session ID
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Generated session ID: {session_id}")
        
        # Extract frames
        print(f"Extracting frames at {self.fps} fps...")
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        frames = self._extract_frames(video_path, frames_dir, duration)
        print(f"Extracted {len(frames)} frames to {frames_dir}")
        
        # Extract and transcribe audio
        transcript = None
        audio_path = None
        
        if extract_audio:
            print("Extracting audio track...")
            audio_path = output_dir / "audio.wav"
            self._extract_audio(video_path, audio_path)
            
            if audio_path.exists():
                audio_size = audio_path.stat().st_size
                print(f"Audio extracted: {audio_path} ({audio_size / 1024:.1f} KB)")
            
            if transcribe and audio_path.exists():
                print("Starting audio transcription...")
                try:
                    transcript = self.transcriber.transcribe_file(audio_path)
                    if transcript and transcript.segments:
                        print(f"Transcription complete: {len(transcript.segments)} segments")
                    else:
                        print("Transcription complete (no speech detected)")
                except Exception as e:
                    logger.warning(f"Transcription failed: {e}")
                    transcript = None
        else:
            print("Audio extraction skipped (extract_audio=False)")
        
        session = ProcessedSession(
            session_id=session_id,
            video_path=video_path,
            output_dir=output_dir,
            frames=frames,
            transcript=transcript,
            audio_path=audio_path,
            duration=duration,
            fps=self.fps,
        )
        
        # Save metadata
        metadata_path = session.save_metadata()
        print(f"Session metadata saved to: {metadata_path}")
        print(f"Video processing complete. Session ID: {session_id}")
        
        return session
    
    def _get_video_duration(self, video_path: Path) -> float:
        """Get the duration of a video file in seconds."""
        print(f"Running ffprobe to get video duration: {video_path}")
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            str(video_path),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ffprobe failed with return code {result.returncode}: {result.stderr}")
            raise RuntimeError(f"ffprobe error: {result.stderr}")
        
        data = json.loads(result.stdout)
        duration = float(data["format"]["duration"])
        print(f"Video duration retrieved: {duration:.2f}s")
        return duration
    
    def _extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        duration: float,
    ) -> list[FrameInfo]:
        """Extract frames from video at specified FPS."""
        # Use ffmpeg to extract frames
        # Output format: frame_NNNN.png with timestamp
        output_pattern = str(output_dir / "frame_%04d.png")
        expected_frames = int(duration * self.fps)
        print(f"Expecting approximately {expected_frames} frames from {duration:.2f}s video at {self.fps} fps")
        
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"fps={self.fps}",
            "-q:v", "2",  # High quality
            output_pattern,
            "-y",  # Overwrite
        ]
        
        print(f"Running ffmpeg frame extraction: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg frame extraction failed: {result.stderr}")
            raise RuntimeError(f"ffmpeg frame extraction error: {result.stderr}")
        
        # Collect extracted frames with timestamps
        frames = []
        frame_paths = sorted(output_dir.glob("frame_*.png"))
        print(f"Found {len(frame_paths)} frame files in {output_dir}")
        
        for i, path in enumerate(frame_paths):
            timestamp = i / self.fps  # Calculate timestamp based on index
            frames.append(FrameInfo(
                path=path,
                timestamp=timestamp,
                index=i,
            ))
        
        if frames:
            total_size = sum(f.path.stat().st_size for f in frames)
            print(f"Total frames size: {total_size / (1024 * 1024):.2f} MB")
        
        return frames
    
    def _extract_audio(self, video_path: Path, output_path: Path) -> None:
        """Extract audio track from video to WAV format."""
        print(f"Extracting audio from {video_path} to {output_path}")
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit
            "-ar", "16000",  # 16kHz sample rate (good for speech)
            "-ac", "1",  # Mono
            str(output_path),
            "-y",  # Overwrite
        ]
        
        print(f"Running ffmpeg audio extraction: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Audio extraction might fail if video has no audio track
            logger.warning(f"Audio extraction failed (video may have no audio track): {result.stderr}")
            # Create empty file to indicate we tried
            output_path.touch()
            print("Created empty audio file as placeholder")
        else:
            print(f"Audio extraction successful: {output_path}")

