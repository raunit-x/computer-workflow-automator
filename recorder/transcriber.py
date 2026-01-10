"""Audio transcription using OpenAI Speech-to-Text API."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI


@dataclass
class TranscriptSegment:
    """A segment of transcribed audio with timing."""
    
    text: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    
    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "TranscriptSegment":
        """Create from dictionary."""
        return cls(
            text=d["text"],
            start=d["start"],
            end=d["end"],
        )


@dataclass
class TranscriptWord:
    """A single word with timing."""
    
    word: str
    start: float
    end: float
    
    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "TranscriptWord":
        """Create from dictionary."""
        return cls(
            word=d["word"],
            start=d["start"],
            end=d["end"],
        )


@dataclass
class Transcript:
    """Complete transcription result."""
    
    text: str
    segments: list[TranscriptSegment]
    words: list[TranscriptWord]
    duration: float
    
    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
            "words": [w.to_dict() for w in self.words],
            "duration": self.duration,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        """Create from dictionary."""
        return cls(
            text=d["text"],
            segments=[TranscriptSegment.from_dict(s) for s in d.get("segments", [])],
            words=[TranscriptWord.from_dict(w) for w in d.get("words", [])],
            duration=d.get("duration", 0.0),
        )
    
    def get_text_at_time(self, timestamp: float, window: float = 2.0) -> str:
        """Get transcribed text near a specific timestamp."""
        relevant_words = [
            w.word for w in self.words
            if w.start >= timestamp - window and w.end <= timestamp + window
        ]
        return " ".join(relevant_words)


class Transcriber:
    """Transcribes audio using OpenAI's Speech-to-Text API."""
    
    def __init__(self, api_key: str | None = None):
        """Initialize the transcriber.
        
        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
        """
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
    
    def transcribe_file(self, audio_path: Path) -> Transcript:
        """Transcribe an audio file.
        
        Args:
            audio_path: Path to the audio file (WAV, MP3, etc.)
            
        Returns:
            Transcript with text, segments, and word-level timestamps.
        """
        with open(audio_path, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
            )
        
        return self._parse_response(response)
    
    def transcribe_bytes(self, audio_data: bytes, filename: str = "audio.wav") -> Transcript:
        """Transcribe audio from bytes.
        
        Args:
            audio_data: Raw audio bytes.
            filename: Filename hint for the API.
            
        Returns:
            Transcript with text, segments, and word-level timestamps.
        """
        # Create a file-like object from bytes
        import io
        audio_file = io.BytesIO(audio_data)
        audio_file.name = filename
        
        response = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )
        
        return self._parse_response(response)
    
    def _parse_response(self, response: Any) -> Transcript:
        """Parse the API response into a Transcript object."""
        segments = []
        if hasattr(response, "segments") and response.segments:
            for seg in response.segments:
                segments.append(TranscriptSegment(
                    text=seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", ""),
                    start=seg.get("start", 0.0) if isinstance(seg, dict) else getattr(seg, "start", 0.0),
                    end=seg.get("end", 0.0) if isinstance(seg, dict) else getattr(seg, "end", 0.0),
                ))
        
        words = []
        if hasattr(response, "words") and response.words:
            for word in response.words:
                words.append(TranscriptWord(
                    word=word.get("word", "") if isinstance(word, dict) else getattr(word, "word", ""),
                    start=word.get("start", 0.0) if isinstance(word, dict) else getattr(word, "start", 0.0),
                    end=word.get("end", 0.0) if isinstance(word, dict) else getattr(word, "end", 0.0),
                ))
        
        duration = getattr(response, "duration", 0.0)
        if not duration and segments:
            duration = max(s.end for s in segments)
        
        return Transcript(
            text=response.text,
            segments=segments,
            words=words,
            duration=duration,
        )

