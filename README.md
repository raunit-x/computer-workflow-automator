# Workflow Automation

**Learn and automate computer workflows from video demonstrations**

A system that observes a human performing a computer task, learns the workflow from that demonstration, and can fully automate the task with new inputs. No code/prompt change change requried.

## Features

- **Video Recording**: Capture screen + audio using macOS screencapture
- **Video Upload**: Analyze existing .mov/.mp4 recordings
- **Multi-Pass Analysis**: Three-pass extraction for accurate workflow understanding
- **Multi-Provider LLM**: Supports Anthropic Claude, OpenAI GPT, and Google Gemini
- **Parameters**: AI identifies and suggests parameterizable inputs
- **Execution**: Replay workflows with new parameters using Claude computer-use API
- **Verification Loop**: Screenshot after each action to verify success
- **UI**: Streamlit interface for all operations
- **Dual Format**: Saves workflows as both .md (human-readable) and .json (machine-parseable)

---

## Architecture

![Architecture Diagram](assets/architecture.png)

### Phase Details

| Phase | Component | Purpose |
|-------|-----------|---------|
| Recording | `recorder/session.py` | Native macOS screen capture with audio |
| Processing | `recorder/video_processor.py` | ffmpeg frame extraction + Whisper transcription |
| Analysis | `analyzer/workflow_extractor.py` | Three-pass LLM extraction pipeline |
| Execution | `executor/loop.py` | Agentic sampling loop with verification |

---

## Design Quality

### Data Models

The system uses well-defined data models for workflows, events, and parameters:

**Workflow Model** (`analyzer/schema.py`):
- YAML frontmatter with metadata (id, name, description, parameters)
- Markdown body with natural language instructions
- Dual serialization: `.md` for humans, `.json` for machines

**Parameter Model**:
```python
class ParameterType(StrEnum):
    STRING = "string"      # Text inputs, search queries
    NUMBER = "number"      # Numeric values
    BOOLEAN = "boolean"    # True/false flags
    SELECTION = "selection" # Dropdown choices
    COORDINATE = "coordinate" # Screen positions
```

**Event Model** (for multi-pass extraction):
```python
@dataclass
class DetectedEvent:
    timestamp: float        # When in the video
    frame_indices: list     # Which frames span this action
    action_type: str        # click, type, scroll, key, etc.
    target: str             # UI element description
    value: str | None       # Text typed, option selected
    intent: str             # WHY the user did this
    before_state: str       # Screen before action
    after_state: str        # Screen after action
    confidence: float       # Detection confidence
```

**Understanding Model** (accumulates across batches):
```python
@dataclass
class RunningUnderstanding:
    task_goal: str              # High-level workflow purpose
    application: str            # Primary app being used
    steps: list[WorkflowStep]   # Accumulated steps
    parameters: list            # Detected parameterizable values
    context_notes: list         # Important observations
    troubleshooting_hints: list # Potential issues
    prerequisites: list         # Required setup
```

### Separation of Concerns

```
workflow_automation/
├── recorder/           # CAPTURE: Screen recording and processing
│   ├── session.py      # macOS screencapture wrapper
│   ├── video_processor.py  # ffmpeg frame/audio extraction
│   └── transcriber.py  # OpenAI Whisper integration
├── analyzer/           # EXTRACT: Video → Workflow conversion
│   ├── schema.py       # Data models (Workflow, Parameter, etc.)
│   ├── workflow_extractor.py  # Multi-pass LLM extraction
│   └── parameter_detector.py  # Parameter identification
├── executor/           # RUN: Workflow execution
│   ├── macos_computer.py  # pyautogui computer tool
│   ├── loop.py         # Agentic sampling loop
│   └── workflow_runner.py  # High-level orchestration
├── prompts/            # LLM prompts (separated from logic)
├── utils/              # Cross-cutting concerns
│   ├── llm.py          # Unified multi-provider LLM client
│   ├── tracking.py     # Cost and time tracking
│   └── logger.py       # Structured logging
└── ui/                 # User interface
    └── streamlit_app.py
```

---

## Accuracy

### Verification Protocol

The execution loop implements explicit verification after each action:

1. **Screenshot After Action**: Every click, type, or key press triggers a screenshot
2. **Visual Verification**: Claude examines the screenshot to confirm success
3. **Explicit Reasoning**: The system prompt requires: *"After each step, take a screenshot and carefully evaluate if you have achieved the right outcome. Explicitly show your thinking: 'I have evaluated step X...'"*
4. **Retry on Failure**: If verification fails, the agent tries alternative approaches

### Adaptability to New Inputs

Workflows use **natural language instructions**, not pixel coordinates:

```markdown
### 2. Search for Restaurants
Navigate to Google and search for: `{search_query}`
```

This allows Claude to:
- Find UI elements by description, not fixed coordinates
- Adapt to different screen sizes and resolutions
- Handle minor UI changes (button moved, different browser)
- Use keyboard shortcuts as fallbacks when clicking fails

### Parameter Substitution

Parameters are substituted at runtime with `{parameter_name}` syntax:

```python
# In workflow_runner.py
full_params = workflow.fill_defaults(parameters)  # Apply defaults
# Claude receives instructions with placeholders to substitute
```

---

## Speed

### Latency Optimizations

| Optimization | Implementation | Impact |
|--------------|----------------|--------|
| **Parallel batch processing** | Gemini requests fire 5 at a time, wait 60s between batches | ~5x faster for Pass 1 |
| **Image compression** | JPEG 80% quality, max 1568px long edge | Smaller API payloads |
| **Frame sampling** | 2 fps extraction (configurable) | 100s video = 200 frames |
| **Cost-optimized routing** | Gemini Flash for Pass 1, Claude for Pass 2/3 | ~10x cheaper detection |

### Model Selection Strategy

```python
@classmethod
def cost_optimized(cls) -> "ModelConfig":
    return cls(
        event_detection="gemini-3-flash-preview",    # Fast, cheap
        understanding="claude-sonnet-4-5-20250929",  # Powerful
        synthesis="claude-sonnet-4-5-20250929",      # Powerful
        execution="claude-sonnet-4-5-20250929",      # Computer-use
    )
```

Pass 1 (Event Detection) processes ~200 frames in chunks of 10, requiring many API calls. Using Gemini Flash at ~$0.50/M tokens vs Claude at ~$3/M tokens provides significant savings.

### Execution Timing

- `action_delay`: 0.3s wait after actions for UI to settle
- `screenshot_delay`: 0.5s before capturing (ensures render complete)
- `typing_interval`: 0.02s between keystrokes (natural speed)

---

## Robustness

### Error Handling & Recovery

**Rate Limiting with Exponential Backoff**:
```python
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 2.0  # seconds
MAX_RETRY_DELAY = 120.0    # seconds

# On 429 error: retry after 2s, 4s, 8s, 16s, 32s...
```

**Overlapping Chunks for Event Detection**:
```python
def _create_overlapping_chunks(frames, chunk_size=10, overlap=2):
    # Chunks overlap by 2 frames to catch actions spanning boundaries
```

**Duplicate Event Deduplication**:
```python
def _merge_events(existing, new, has_overlap):
    # Skip duplicates by comparing timestamp + action_type + target
    for event in new_events:
        if abs(event.timestamp - existing.timestamp) < 0.5 and
           event.action_type == existing.action_type:
            is_duplicate = True
```

**Coordinate Scaling**:
```python
# Screenshots are resized to meet API constraints (max 1568px, ~1.15MP)
# Coordinates from Claude are scaled back to actual screen dimensions
def scale_coordinates(self, source: ScalingSource, x, y):
    if source == ScalingSource.API:
        return round(x / self._scale_factor), round(y / self._scale_factor)
```

**Safety Mechanisms**:
```python
pyautogui.FAILSAFE = True   # Move mouse to corner to abort
pyautogui.PAUSE = 0.1       # Small pause between actions
```

**Graceful Degradation**:
- If audio extraction fails (no audio track), continues without transcript
- If JSON parsing fails, falls back to previous understanding
- If transcription fails, workflow still generates from visual analysis

**Reference Frames Fallback**:
When the agent gets stuck during execution, it can access the original recording frames as a fallback:
- **Auto-trigger**: After 3 consecutive failed attempts on the same action
- **Agent-requested**: Claude can call `reference_frames(step_number=N)` when stuck
- Shows what the screen looked like when the human performed that step
- Helps identify correct UI elements when the interface has changed

```python
# Agent can request reference frames when stuck
reference_frames(step_number=5)   # Get frames for step 5
reference_frames(timestamp=30.0)  # Get frames near 30s mark
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Recording format** | Video file (.mov) | Use existing tools, easy to review/share |
| **Frame extraction** | ffmpeg at 2 fps | Reliable, cross-platform, configurable |
| **Voice transcription** | OpenAI Whisper API | Word-level timestamps, high accuracy |
| **Workflow format** | Both .md and .json | Human-editable + machine-parseable |
| **Instructions** | Natural language | Claude interprets intent, not just actions |
| **Execution model** | Claude computer-use | Handles UI variance gracefully |
| **Multi-pass extraction** | 3 passes | Handles long videos, builds context incrementally |
| **Model routing** | Gemini Flash + Claude Sonnet | Cost optimization (~10x cheaper for Pass 1) |
| **LLM abstraction** | Unified LLMClient | Provider-agnostic, easy to swap models |
| **Verification loop** | Screenshot after action | Ensures each step succeeded before proceeding |
| **Reference frames fallback** | Original recording frames | Helps agent recover when stuck on a step |
| **Image format** | JPEG 80% quality | Balances quality vs API payload size |
| **Package manager** | uv | Fast, modern Python tooling |

---

## Limitations

- **macOS only**: Uses macOS-specific features (screencapture, pyautogui)
- **Accessibility permissions**: Must manually grant terminal permissions
- **Network required**: API calls to Anthropic, OpenAI, and optionally Google
- **Expensive**: A 3 min workflow with sonnet 4.5 would cost ~$10-20

---

## Setup

### Prerequisites

- macOS (for the computer control features)
- Python 3.11-3.13
- [uv](https://docs.astral.sh/uv/) package manager
- **ffmpeg** (for video processing): `brew install ffmpeg`
- Anthropic API key (required for execution)
- OpenAI API key (required for voice transcription)
- Google API key (optional, for cost-optimized analysis)

### Installation

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install ffmpeg
brew install ffmpeg

# Navigate to the project
cd workflow_automation

# Install dependencies
uv sync

# Verify installation
uv run python main.py --help
```

### macOS Permissions Required

1. **Accessibility**: Required for mouse/keyboard control
   - System Preferences > Security & Privacy > Privacy > Accessibility
   - Add your terminal (Terminal, iTerm, VS Code, etc.)

2. **Screen Recording**: Required for screen capture
   - System Preferences > Security & Privacy > Privacy > Screen Recording
   - Add your terminal

3. **Microphone** (for audio recording):
   - System Preferences > Security & Privacy > Privacy > Microphone
   - Add your terminal

### Environment Variables

```bash
# Required
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"

# Optional (for cost-optimized analysis using Gemini Flash)
export GOOGLE_API_KEY="your-google-key"
```

> **Note**: If `GOOGLE_API_KEY` is not set, analysis will use Claude for all passes (works fine, but costs more). The Gemini free tier has rate limits (5 requests/minute), which the system handles with automatic batching and backoff.

---

## Usage

### Streamlit UI (Recommended)

```bash
uv run python main.py ui
```

This opens a web interface with three tabs:
1. **Record & Analyze**: Record screen or upload video, analyze to extract workflow
2. **Review**: View/edit extracted workflows and parameters
3. **Execute**: Run workflows with new parameters

### Command Line

```bash
# Record a screen recording (saves .mov file)
uv run python main.py record --output ./recordings

# Analyze a video file → generates both .md and .json
uv run python main.py analyze ./recordings/recording_20240107_120000.mov

# Analyze with specific model options
uv run python main.py analyze video.mov \
    --event-model gemini-3-flash-preview \
    --synthesis-model claude-sonnet-4-5-20250929

# List available workflows
uv run python main.py list

# Show workflow details
uv run python main.py show workflows/restaurant_search.md

# Run a workflow with new parameters
uv run python main.py run workflows/restaurant_search.md \
    -p search_query="best pizza restaurants in San Jose"
```

---

## How It Works

### 1. Recording

Use macOS screencapture or your preferred screen recorder:
- The `record` command uses `screencapture -v -g -k` for native macOS recording
- You can also record with QuickTime, OBS, or any tool that produces .mov/.mp4

### 2. Video Processing

When you run `analyze`:
- ffmpeg extracts frames at 2 fps (1 frame every 500ms)
- ffmpeg extracts the audio track to WAV
- OpenAI Whisper API transcribes the audio with timestamps
- Frames and transcript are packaged as a `ProcessedSession`

### 3. Multi-Pass Workflow Extraction

**Pass 1 - Event Detection** (Gemini Flash):
- Frames analyzed in overlapping chunks of 10
- Detects discrete user actions: clicks, typing, scrolling, navigation
- Outputs list of `DetectedEvent` objects with timestamps and intent

**Pass 2 - Understanding Building** (Claude Sonnet):
- Events processed in batches of 15
- Incrementally builds `RunningUnderstanding` with steps, parameters, context
- Merges related actions into cohesive workflow steps

**Pass 3 - Workflow Synthesis** (Claude Sonnet):
- Complete understanding → polished markdown workflow
- Generates YAML frontmatter with parameters
- Adds troubleshooting hints and context notes

### 4. Execution

When running a workflow:
- Claude receives the markdown instructions and new parameters
- Takes an initial screenshot to see the current screen state
- For each step:
  1. Execute action using `MacOSComputerTool`
  2. Take screenshot to verify result
  3. Evaluate success and adapt if needed
- Adapts to UI variations using natural language understanding

---

## Workflow Format

Workflows are stored in **two formats**:

### Markdown (.md) - Human-readable

```markdown
---
id: restaurant_search
name: "Restaurant Search and Note"
description: "Search for restaurants and save to Notes"
parameters:
  - name: search_query
    type: string
    description: "The search query for restaurants"
    default: "best sushi restaurants in San Francisco"
    required: true
---

# Restaurant Search Workflow

## Goal
Search for restaurants matching a query and save the results.

## Important Context
- Uses Safari browser on macOS
- Keyboard shortcuts are macOS-specific (Cmd instead of Ctrl)

## Steps

### 1. Open Browser
Press **Cmd+Space** to open Spotlight, type "Safari", press Enter.

### 2. Search for Restaurants
Navigate to Google and search for: `{search_query}`

## Troubleshooting
- If Spotlight doesn't open, try clicking the magnifying glass
```

### JSON (.json) - Machine-parseable

```json
{
  "id": "restaurant_search",
  "name": "Restaurant Search and Note",
  "description": "Search for restaurants and save to Notes",
  "parameters": [
    {
      "name": "search_query",
      "type": "string",
      "description": "The search query for restaurants",
      "default": "best sushi restaurants in San Francisco",
      "required": true
    }
  ],
  "instructions": "# Restaurant Search Workflow\n\n..."
}
```

---

## Project Structure

```
workflow_automation/
├── pyproject.toml           # uv project config with dependencies
├── config.py                # ModelConfig and application settings
├── main.py                  # CLI entry point (Typer)
├── recorder/
│   ├── session.py           # Screen recording (screencapture)
│   ├── video_processor.py   # Extract frames/audio (ffmpeg)
│   └── transcriber.py       # OpenAI Speech-to-Text API
├── analyzer/
│   ├── schema.py            # Workflow, Parameter, Event data models
│   ├── workflow_extractor.py# Multi-pass LLM extraction
│   ├── parameter_detector.py# Additional parameter detection
│   └── json_utils.py        # JSON parsing utilities
├── executor/
│   ├── macos_computer.py    # macOS computer control tool
│   ├── loop.py              # Agentic sampling loop
│   └── workflow_runner.py   # High-level workflow runner
├── prompts/
│   ├── analyzer_prompts.py  # Extraction prompts (3 passes)
│   └── executor_prompts.py  # Execution system prompt
├── utils/
│   ├── llm.py               # Unified LLMClient (Anthropic/OpenAI/Gemini)
│   ├── tracking.py          # CostTracker, Timer
│   └── logger.py            # WorkflowLogger
├── ui/
│   └── streamlit_app.py     # Streamlit UI
├── recordings/              # Screen recordings (.mov)
├── processed/               # Extracted frames and audio
├── workflows/               # Workflow definitions (.md + .json)
└── logs/                    # Execution logs
```

---

## Troubleshooting

### "ffmpeg not found"

Install ffmpeg:
```bash
brew install ffmpeg
```

### "Permission denied" for screen capture

Grant accessibility permissions:
1. System Preferences > Security & Privacy > Privacy
2. Select "Accessibility" in the left panel
3. Click the lock to make changes
4. Add your terminal application

### Audio not transcribed

1. Ensure your video has an audio track
2. Check that `OPENAI_API_KEY` is set
3. The audio might be silent - try narrating while recording

### Rate limit errors with Gemini

The system automatically handles Gemini rate limits with exponential backoff. If you see many retries:
- Wait a few minutes and try again
- Or set `--event-model claude-sonnet-4-5-20250929` to use Claude for all passes

### pyautogui fails to click

Ensure the terminal has accessibility permissions and try:
```bash
uv run python -c "import pyautogui; print(pyautogui.position())"
```