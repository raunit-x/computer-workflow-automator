# Workflow Automation

**Learn and automate computer workflows from video demonstrations**

A system that observes a human performing a computer task, learns the workflow from that demonstration, and can fully automate the task with new inputs. Trainable by any user simply by recording a new demo — no code changes required.

## Features

- **ideo Recording**: Capture screen + audio using macOS screencapture
- **Video Upload**: Analyze existing .mov/.mp4 recordings
- **Analysis**: Claude extracts structured workflows with natural language instructions
- **Parameters**: AI identifies and suggests parameterizable inputs
- **Execution**: Replay workflows with new parameters using AI-guided computer control
- **UI**: Beautiful Streamlit interface for all operations
- **Dual Format**: Saves workflows as both .md (human-readable) and .json (machine-parseable)

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Recording Phase                              │
│                                                                      │
│    macOS screencapture -v -A -k output.mov                           │
│    • Screen capture                                                  │
│    • Audio recording (microphone)                                    │
│    • Mouse click visualization                                       │
│                                                                      │
│                         ┌───────────────┐                            │
│                         │  video.mov    │                            │
│                         └───────┬───────┘                            │
└─────────────────────────────────┼────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      Video Processing Phase                          │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                   VideoProcessor (ffmpeg)                       │ │
│  │                                                                 │ │
│  │  video.mov ─┬─► Extract frames (2 fps) ──► frames/*.png         │ │
│  │             └─► Extract audio ──► audio.wav                     │ │
│  │                                                                 │ │
│  │  audio.wav ───► OpenAI Whisper API ──► Transcript               │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                  │                                   │
│                                  ▼                                   │
│                        ProcessedSession                              │
│                    (frames + transcript)                             │
└─────────────────────────────────┼────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         Analysis Phase                               │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │              Claude Workflow Extractor                          │ │
│  │                                                                 │ │
│  │  • Frames with timestamps                                       │ │
│  │  • Voice transcription with timestamps                          │ │
│  │  ──────────────────────────────────────────────────────────►    │ │
│  │  • Workflow with natural language instructions                  │ │
│  │  • Identified parameters                                        │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│                              ▼                                       │
│                   ┌─────────────────────┐                            │
│                   │ workflow.md         │ (human-readable)           │
│                   │ workflow.json       │ (machine-parseable)        │
│                   └─────────────────────┘                            │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Execution Phase                              │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │ New Parameters   │  │ Claude Executor  │  │ macOS Computer   │   │
│  │ (user input)     │──│ (sampling loop)  │──│ Tool (pyautogui) │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘   │
│                              │                        │             │
│                              └────────────────────────┘             │
│                                   Screenshot feedback               │
└─────────────────────────────────────────────────────────────────────┘
```

## Workflow Format

Workflows are stored in **two formats**:

### 1. Markdown (.md) - Human-readable

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

*The {search_query} parameter will be substituted with the actual value.*

### 3. Save Results
Open Notes app and save the restaurant information.

## Troubleshooting
- If Spotlight doesn't open, try clicking the magnifying glass
```

### 2. JSON (.json) - Machine-parseable

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

## Project Structure

```
workflow_automation/
├── pyproject.toml           # uv project config
├── config.py                # Configuration settings
├── main.py                  # CLI entry point
├── recorder/
│   ├── session.py           # Screen recording (screencapture)
│   ├── video_processor.py   # Extract frames/audio (ffmpeg)
│   └── transcriber.py       # OpenAI Speech-to-Text API
├── analyzer/
│   ├── schema.py            # Workflow data models
│   ├── workflow_extractor.py# Claude-based workflow generation
│   └── parameter_detector.py# Parameter detection
├── executor/
│   ├── macos_computer.py    # macOS computer control tool
│   ├── loop.py              # Agentic sampling loop
│   └── workflow_runner.py   # High-level workflow runner
├── ui/
│   └── streamlit_app.py     # Streamlit UI
├── recordings/              # Screen recordings (.mov)
├── processed/               # Extracted frames and audio
└── workflows/               # Workflow definitions (.md + .json)
```

## Setup

### Prerequisites

- macOS (for the computer control features)
- Python 3.11-3.13
- [uv](https://docs.astral.sh/uv/) package manager
- **ffmpeg** (for video processing): `brew install ffmpeg`
- Anthropic API key
- OpenAI API key (for voice transcription)

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
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
```

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

# Or analyze any video file
uv run python main.py analyze ~/Desktop/my_demo.mp4 --output workflows/my_workflow

# List available workflows
uv run python main.py list

# Show workflow details
uv run python main.py show workflows/restaurant_search.md

# Run a workflow with new parameters
uv run python main.py run workflows/restaurant_search.md \
    -p search_query="best pizza restaurants in San Jose"
```

## How It Works

### 1. Recording

Use macOS screencapture or your preferred screen recorder:
- The `record` command uses `screencapture -v -g -k` for native macOS recording
- You can also record with QuickTime, OBS, or any tool that produces .mov/.mp4

### 2. Video Processing

When you run `analyze`:
- ffmpeg extracts frames at 2 fps (1 frame every 500ms)
- ffmpeg extracts the audio track
- OpenAI Whisper API transcribes the audio with timestamps
- Frames and transcript are packaged for analysis

### 3. Workflow Extraction

Claude receives:
- Screenshots with timestamps showing what was on screen
- Voice-over transcription with timestamps

Claude generates:
- Comprehensive markdown workflow with natural language instructions
- YAML frontmatter with metadata and parameters
- Reasoning, context, and troubleshooting tips

### 4. Execution

When running a workflow:
- Claude receives the markdown instructions and new parameters
- Takes an initial screenshot to see the current screen state
- Executes the workflow using visual understanding
- Adapts to UI changes using the natural language instructions
- Takes screenshots after each action for verification

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Recording | Video file (.mov) | Use existing tools, easy to review/share |
| Frame extraction | ffmpeg (2 fps) | Reliable, cross-platform, efficient |
| Voice transcription | OpenAI Whisper API | Word-level timestamps, high accuracy |
| Workflow format | Both .md and .json | Human-editable + machine-parseable |
| Instructions | Natural language | Claude interprets intent, not just actions |
| Execution model | Claude-guided | Handles UI variance gracefully |
| Package manager | uv | Fast, modern Python tooling |

## Limitations

- **macOS only**: Currently uses macOS-specific features (screencapture, pyautogui)
- **ffmpeg required**: Must install ffmpeg for video processing
- **Accessibility permissions required**: Must grant terminal accessibility access
- **Network required**: API calls to Anthropic and OpenAI
- **Single monitor**: Multi-monitor support not yet implemented

## Possible Extensions

- Windows/Linux support
- Multi-monitor handling
- Workflow branching (if/else based on screen state)
- Workflow composition (chain multiple workflows)
- Cloud execution mode
- Workflow versioning and diff
- Collaborative workflow sharing
- Local Whisper model option

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
2. Check that OPENAI_API_KEY is set
3. The audio might be silent - try narrating while recording

### pyautogui fails to click

Ensure the terminal has accessibility permissions and try:
```bash
uv run python -c "import pyautogui; print(pyautogui.position())"
```

## License

See LICENSE file in the parent directory.
