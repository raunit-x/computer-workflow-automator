"""Recording session using macOS screencapture for video recording."""

import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class RecordingSession:
    """Records screen and audio using macOS screencapture command.
    
    Uses: screencapture -v -g -k output.mov
      -v = video mode
      -g = capture audio (microphone) during video recording
      -k = show clicks in video recording mode
    """
    
    output_dir: Path
    capture_clicks: bool = True
    capture_audio: bool = True
    
    _process: subprocess.Popen | None = field(default=None, init=False)
    _start_time: float = field(default=0.0, init=False)
    _video_path: Path | None = field(default=None, init=False)
    _running: bool = field(default=False, init=False)
    
    def __post_init__(self):
        """Initialize the recording session."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def video_path(self) -> Path:
        """Get the path to the output video file."""
        if self._video_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._video_path = self.output_dir / f"recording_{timestamp}.mov"
        return self._video_path
    
    def start(self) -> None:
        """Start screen recording."""
        if self._running:
            return
        
        # Build screencapture command
        cmd = ["screencapture", "-v"]
        
        if self.capture_audio:
            cmd.append("-g")  # Capture audio during video recording
        
        if self.capture_clicks:
            cmd.append("-k")  # Show clicks in video recording mode
        
        cmd.append(str(self.video_path))
        
        # Start screencapture in background
        # Note: screencapture -v runs until interrupted with Ctrl+C
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setpgrp,  # Create new process group
        )
        
        self._start_time = time.time()
        self._running = True
    
    def stop(self) -> Path:
        """Stop screen recording and return the video path.
        
        Returns:
            Path to the recorded video file.
        """
        if not self._running or self._process is None:
            raise RuntimeError("Recording is not running")
        
        # Send Ctrl+C (SIGINT) to stop screencapture gracefully
        try:
            # Send to process group to ensure child processes also receive it
            os.killpg(os.getpgid(self._process.pid), signal.SIGINT)
            
            # Wait for process to finish (with timeout)
            self._process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't stop
            os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
            self._process.wait()
        except ProcessLookupError:
            # Process already terminated
            pass
        
        self._running = False
        
        # Give a moment for file to be finalized
        time.sleep(0.5)
        
        return self.video_path
    
    @property
    def is_recording(self) -> bool:
        """Check if recording is in progress."""
        return self._running
    
    @property
    def duration(self) -> float:
        """Get current recording duration in seconds."""
        if not self._running:
            return 0.0
        return time.time() - self._start_time


def record_video(
    output_path: Path,
    capture_audio: bool = True,
    capture_clicks: bool = True,
) -> Path:
    """Convenience function to record a video.
    
    This is a blocking function that records until interrupted with Ctrl+C.
    
    Args:
        output_path: Path for the output video file.
        capture_audio: Whether to capture microphone audio.
        capture_clicks: Whether to show mouse clicks in recording.
        
    Returns:
        Path to the recorded video file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = ["screencapture", "-v"]
    
    if capture_audio:
        cmd.append("-g")  # Capture audio during video recording
    
    if capture_clicks:
        cmd.append("-k")
    
    cmd.append(str(output_path))
    
    print(f"Starting recording: {output_path}")
    print("Press Ctrl+C to stop recording...")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nRecording stopped.")
    
    return output_path
