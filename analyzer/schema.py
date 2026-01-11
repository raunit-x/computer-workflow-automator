"""Data models for workflows, steps, and parameters."""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


# =============================================================================
# Multi-Pass Video Analysis Data Structures
# =============================================================================


@dataclass
class DetectedEvent:
    """An event/action detected from video frames during Pass 1.
    
    Represents a discrete user action identified from analyzing frames,
    including context about what happened before and after.
    """
    
    timestamp: float  # Time in seconds from video start
    frame_indices: list[int]  # Frames this event spans (for multi-frame actions)
    action_type: str  # click, type, scroll, navigate, wait, select, drag
    target: str  # Human-readable description of UI element
    value: str | None  # Typed text, clicked item, scroll direction, etc.
    intent: str  # Inferred purpose/reason for this action
    before_state: str  # Description of screen state before action
    after_state: str  # Description of screen state after action
    confidence: float = 1.0  # Confidence score (0.0 to 1.0)
    metadata: dict = field(default_factory=dict)  # Additional context
    
    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "timestamp": self.timestamp,
            "frame_indices": self.frame_indices,
            "action_type": self.action_type,
            "target": self.target,
            "value": self.value,
            "intent": self.intent,
            "before_state": self.before_state,
            "after_state": self.after_state,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "DetectedEvent":
        """Create from dictionary."""
        return cls(
            timestamp=d["timestamp"],
            frame_indices=d.get("frame_indices", []),
            action_type=d["action_type"],
            target=d["target"],
            value=d.get("value"),
            intent=d.get("intent", ""),
            before_state=d.get("before_state", ""),
            after_state=d.get("after_state", ""),
            confidence=d.get("confidence", 1.0),
            metadata=d.get("metadata", {}),
        )


@dataclass
class DetectedParameter:
    """A parameterizable value detected during analysis."""
    
    name: str  # Suggested parameter name
    value: str  # The actual value observed in the recording
    param_type: str  # string, number, boolean, selection
    description: str  # What this parameter represents
    source_events: list[int]  # Indices of events where this was detected
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "param_type": self.param_type,
            "description": self.description,
            "source_events": self.source_events,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "DetectedParameter":
        return cls(
            name=d["name"],
            value=d["value"],
            param_type=d.get("param_type", "string"),
            description=d.get("description", ""),
            source_events=d.get("source_events", []),
        )


@dataclass
class WorkflowStep:
    """A step in the workflow being built during Pass 2."""
    
    step_number: int
    title: str
    description: str
    action_type: str
    target: str
    details: str
    related_events: list[int]  # Indices of DetectedEvents
    voice_context: str | None = None  # Relevant transcript content
    reference_frame_paths: list[str] = field(default_factory=list)  # Paths to frames for this step
    reference_frame_timestamps: list[float] = field(default_factory=list)  # Timestamps of frames
    
    def to_dict(self) -> dict:
        return {
            "step_number": self.step_number,
            "title": self.title,
            "description": self.description,
            "action_type": self.action_type,
            "target": self.target,
            "details": self.details,
            "related_events": self.related_events,
            "voice_context": self.voice_context,
            "reference_frame_paths": self.reference_frame_paths,
            "reference_frame_timestamps": self.reference_frame_timestamps,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "WorkflowStep":
        return cls(
            step_number=d["step_number"],
            title=d["title"],
            description=d.get("description", ""),
            action_type=d.get("action_type", ""),
            target=d.get("target", ""),
            details=d.get("details", ""),
            related_events=d.get("related_events", []),
            voice_context=d.get("voice_context"),
            reference_frame_paths=d.get("reference_frame_paths", []),
            reference_frame_timestamps=d.get("reference_frame_timestamps", []),
        )


@dataclass
class RunningUnderstanding:
    """Accumulated understanding built during Pass 2.
    
    This state is incrementally refined as we process batches of events,
    building up a complete picture of the workflow.
    """
    
    task_goal: str  # High-level description of what the workflow accomplishes
    application: str  # Primary application being used
    steps: list[WorkflowStep]  # Accumulated workflow steps
    parameters: list[DetectedParameter]  # Detected parameterizable values
    context_notes: list[str]  # Important observations about the workflow
    troubleshooting_hints: list[str]  # Potential issues and solutions
    prerequisites: list[str]  # Required setup or conditions
    current_screen_context: str  # Latest screen state for continuity
    events_processed: int  # Number of events processed so far
    
    def to_dict(self) -> dict:
        return {
            "task_goal": self.task_goal,
            "application": self.application,
            "steps": [s.to_dict() for s in self.steps],
            "parameters": [p.to_dict() for p in self.parameters],
            "context_notes": self.context_notes,
            "troubleshooting_hints": self.troubleshooting_hints,
            "prerequisites": self.prerequisites,
            "current_screen_context": self.current_screen_context,
            "events_processed": self.events_processed,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "RunningUnderstanding":
        return cls(
            task_goal=d.get("task_goal", ""),
            application=d.get("application", ""),
            steps=[WorkflowStep.from_dict(s) for s in d.get("steps", [])],
            parameters=[DetectedParameter.from_dict(p) for p in d.get("parameters", [])],
            context_notes=d.get("context_notes", []),
            troubleshooting_hints=d.get("troubleshooting_hints", []),
            prerequisites=d.get("prerequisites", []),
            current_screen_context=d.get("current_screen_context", ""),
            events_processed=d.get("events_processed", 0),
        )
    
    @classmethod
    def empty(cls) -> "RunningUnderstanding":
        """Create an empty initial state."""
        return cls(
            task_goal="",
            application="",
            steps=[],
            parameters=[],
            context_notes=[],
            troubleshooting_hints=[],
            prerequisites=[],
            current_screen_context="",
            events_processed=0,
        )


# =============================================================================
# Core Workflow Types
# =============================================================================


class ParameterType(StrEnum):
    """Types of parameters."""
    
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    SELECTION = "selection"  # One of predefined options
    COORDINATE = "coordinate"


@dataclass
class Parameter:
    """A parameterizable input in a workflow."""
    
    name: str
    param_type: ParameterType = ParameterType.STRING
    description: str = ""
    default_value: Any = None
    required: bool = True
    options: list[str] | None = None  # For SELECTION type
    
    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "name": self.name,
            "type": self.param_type.value,
            "description": self.description,
            "default": self.default_value,
            "required": self.required,
            "options": self.options,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Parameter":
        """Create from dictionary."""
        # Support both 'type' and 'param_type' keys
        param_type_str = d.get("type") or d.get("param_type", "string")
        
        # Handle unknown types gracefully by falling back to STRING
        try:
            param_type = ParameterType(param_type_str)
        except ValueError:
            # Unknown type (e.g., "date", "datetime", etc.) - fall back to string
            param_type = ParameterType.STRING
        
        return cls(
            name=d["name"],
            param_type=param_type,
            description=d.get("description", ""),
            default_value=d.get("default") or d.get("default_value"),
            required=d.get("required", True),
            options=d.get("options"),
        )


@dataclass
class Workflow:
    """A complete workflow that can be executed.
    
    Supports both JSON and Markdown formats:
    - JSON: Structured step-by-step format (legacy)
    - Markdown: YAML frontmatter + natural language instructions (preferred)
    """
    
    id: str
    name: str
    description: str
    parameters: list[Parameter]
    instructions: str  # The markdown body with detailed instructions
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source_session_id: str | None = None
    processed_session_path: str | None = None  # Path to processed/ folder with reference frames
    
    def to_markdown(self) -> str:
        """Convert to markdown format with YAML frontmatter."""
        # Build YAML frontmatter
        frontmatter = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
        }
        
        if self.source_session_id:
            frontmatter["source_session_id"] = self.source_session_id
        
        if self.processed_session_path:
            frontmatter["processed_session_path"] = self.processed_session_path
        
        if self.parameters:
            frontmatter["parameters"] = [p.to_dict() for p in self.parameters]
        
        yaml_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        
        return f"---\n{yaml_str}---\n\n{self.instructions}"
    
    @classmethod
    def from_markdown(cls, content: str) -> "Workflow":
        """Parse workflow from markdown with YAML frontmatter."""
        # Split frontmatter and body
        frontmatter, instructions = cls._parse_frontmatter(content)
        
        # Parse parameters
        parameters = []
        for p in frontmatter.get("parameters", []):
            parameters.append(Parameter.from_dict(p))
        
        return cls(
            id=frontmatter.get("id", "workflow"),
            name=frontmatter.get("name", "Untitled Workflow"),
            description=frontmatter.get("description", ""),
            parameters=parameters,
            instructions=instructions.strip(),
            created_at=frontmatter.get("created_at", datetime.now().isoformat()),
            source_session_id=frontmatter.get("source_session_id"),
            processed_session_path=frontmatter.get("processed_session_path"),
        )
    
    @staticmethod
    def _parse_frontmatter(content: str) -> tuple[dict, str]:
        """Parse YAML frontmatter from markdown content."""
        # Match --- at start, then content, then ---
        pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
        match = re.match(pattern, content, re.DOTALL)
        
        if match:
            yaml_content = match.group(1)
            body = match.group(2)
            try:
                frontmatter = yaml.safe_load(yaml_content) or {}
            except yaml.YAMLError:
                frontmatter = {}
            return frontmatter, body
        
        # No frontmatter found, treat entire content as instructions
        return {}, content
    
    def save(self, path: Path) -> None:
        """Save workflow to file (markdown or JSON based on extension)."""
        path = Path(path)
        
        if path.suffix == ".md":
            with open(path, "w") as f:
                f.write(self.to_markdown())
        else:
            # JSON format
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
    
    def save_both(self, base_path: Path) -> tuple[Path, Path]:
        """Save workflow to both .md and .json formats.
        
        Args:
            base_path: Base path without extension (e.g., 'workflows/my_workflow')
            
        Returns:
            Tuple of (md_path, json_path)
        """
        base_path = Path(base_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        md_path = base_path.with_suffix(".md")
        json_path = base_path.with_suffix(".json")
        
        self.save(md_path)
        self.save(json_path)
        
        return md_path, json_path
    
    def save_json(self, path: Path) -> None:
        """Save workflow to JSON format.
        
        Args:
            path: Path to save the JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "Workflow":
        """Load workflow from file (supports both .md and .json)."""
        path = Path(path)
        
        with open(path) as f:
            content = f.read()
        
        if path.suffix == ".md":
            return cls.from_markdown(content)
        else:
            # Legacy JSON format
            return cls.from_dict(json.loads(content))
    
    def to_dict(self) -> dict:
        """Convert to dictionary (for JSON serialization)."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "instructions": self.instructions,
            "created_at": self.created_at,
            "source_session_id": self.source_session_id,
            "processed_session_path": self.processed_session_path,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Workflow":
        """Create from dictionary (supports legacy JSON format)."""
        # Handle legacy format with 'steps' instead of 'instructions'
        if "steps" in d and "instructions" not in d:
            # Convert legacy steps to instructions markdown
            instructions = cls._convert_legacy_steps(d.get("steps", []))
        else:
            instructions = d.get("instructions", "")
        
        parameters = []
        for p in d.get("parameters", []):
            parameters.append(Parameter.from_dict(p))
        
        return cls(
            id=d.get("id", "workflow"),
            name=d.get("name", "Untitled Workflow"),
            description=d.get("description", ""),
            parameters=parameters,
            instructions=instructions,
            created_at=d.get("created_at", datetime.now().isoformat()),
            source_session_id=d.get("source_session_id"),
            processed_session_path=d.get("processed_session_path"),
        )
    
    @staticmethod
    def _convert_legacy_steps(steps: list[dict]) -> str:
        """Convert legacy JSON steps to markdown instructions."""
        lines = ["# Workflow Steps\n"]
        
        for step in steps:
            order = step.get("order", 0)
            desc = step.get("description", "")
            action = step.get("action_type", "")
            target = step.get("target_description", "")
            hints = step.get("hints", {})
            voice = step.get("voice_context", "")
            params = step.get("parameters", [])
            
            lines.append(f"## Step {order}: {desc}\n")
            lines.append(f"**Action:** `{action}`\n")
            lines.append(f"**Target:** {target}\n")
            
            if hints.get("text"):
                lines.append(f"**Text/Key:** `{hints['text']}`\n")
            if hints.get("coordinates"):
                lines.append(f"**Coordinates:** {hints['coordinates']}\n")
            if hints.get("element_description"):
                lines.append(f"**Element:** {hints['element_description']}\n")
            
            if params:
                lines.append("\n**Parameters:**\n")
                for p in params:
                    lines.append(f"- `{{{p['name']}}}`: {p.get('description', '')} (default: `{p.get('default_value', '')}`)\n")
            
            if voice:
                lines.append(f"\n*Voice context: {voice}*\n")
            
            lines.append("\n")
        
        return "".join(lines)
    
    def get_all_parameters(self) -> list[Parameter]:
        """Get all parameters."""
        return list(self.parameters)
    
    def validate_parameters(self, provided_params: dict) -> list[str]:
        """Validate that required parameters are provided. Returns list of errors."""
        errors = []
        for param in self.parameters:
            if param.required and param.name not in provided_params:
                if param.default_value is None:
                    errors.append(f"Required parameter '{param.name}' not provided")
        return errors
    
    def fill_defaults(self, provided_params: dict) -> dict:
        """Fill in default values for missing parameters."""
        result = dict(provided_params)
        for param in self.parameters:
            if param.name not in result and param.default_value is not None:
                result[param.name] = param.default_value
        return result
