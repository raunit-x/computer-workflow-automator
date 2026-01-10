"""Streamlit UI for workflow recording, review, and execution."""

import asyncio
import base64
import json
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from anthropic.types.beta import BetaContentBlockParam

# Add parent to path for imports
import sys
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from config import get_config, update_config
from analyzer.schema import Workflow, Parameter, ParameterType
from analyzer.workflow_extractor import WorkflowExtractor
from analyzer.parameter_detector import ParameterDetector
from executor.workflow_runner import WorkflowRunner
from executor.macos_computer import MacOSComputerTool, ToolResult


# Page config
st.set_page_config(
    page_title="Workflow Automation",
    page_icon="🤖",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        font-weight: 600;
    }
    .recording-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: #ff4444;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .param-card {
        background-color: #e3f2fd;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .upload-area {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)


def get_workflow_files(workflows_dir: Path) -> list[Path]:
    """Get all workflow files (.md and .json) from directory."""
    files = []
    if workflows_dir.exists():
        files.extend(workflows_dir.glob("*.md"))
        files.extend(workflows_dir.glob("*.json"))
    return sorted(files, key=lambda p: p.stem)


def setup_state():
    """Initialize session state."""
    if "config" not in st.session_state:
        st.session_state.config = get_config()
    
    if "current_workflow" not in st.session_state:
        st.session_state.current_workflow = None
    
    if "execution_messages" not in st.session_state:
        st.session_state.execution_messages = []
    
    if "execution_running" not in st.session_state:
        st.session_state.execution_running = False
    
    if "analyzing" not in st.session_state:
        st.session_state.analyzing = False


def render_sidebar():
    """Render the sidebar with settings."""
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # API Keys
        st.subheader("API Keys")
        
        anthropic_key = st.text_input(
            "Anthropic API Key",
            value=os.getenv("ANTHROPIC_API_KEY", ""),
            type="password",
            help="Your Anthropic API key for Claude"
        )
        if anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        
        openai_key = st.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="Your OpenAI API key for speech transcription"
        )
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        
        st.divider()
        
        # Model settings
        st.subheader("Model")
        
        model = st.selectbox(
            "Claude Model",
            options=[
                "claude-sonnet-4-5-20250929",
                "claude-sonnet-4-20250514",
                "claude-opus-4-20250514",
            ],
            index=0,
        )
        update_config(model=model)
        
        st.divider()
        
        # Storage info
        st.subheader("Storage")
        recordings_dir = Path("./recordings")
        workflows_dir = Path("./workflows")
        
        st.caption(f"📁 Recordings: {recordings_dir.absolute()}")
        st.caption(f"📁 Workflows: {workflows_dir.absolute()}")
        
        st.divider()
        
        # Requirements
        st.subheader("Requirements")
        
        # Check ffmpeg
        ffmpeg_available = shutil.which("ffmpeg") is not None
        if ffmpeg_available:
            st.success("✅ ffmpeg installed")
        else:
            st.error("❌ ffmpeg not found")
            st.caption("Install with: `brew install ffmpeg`")


def render_record_tab():
    """Render the recording and analysis tab."""
    st.header("🎬 Record & Analyze")
    
    st.markdown("""
    Record your workflow demonstration or upload an existing video file.
    The system will analyze the recording and extract a structured workflow.
    """)
    
    # Two columns: Record or Upload
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📹 Record New")
        st.markdown("""
        Use the command line to record your screen:
        
        ```bash
        python main.py record
        ```
        
        This will:
        - Record your screen and audio
        - Show mouse clicks in the recording
        - Save as `.mov` file
        
        Press `Ctrl+C` to stop recording.
        """)
    
    with col2:
        st.subheader("📤 Upload Video")
        
        uploaded_file = st.file_uploader(
            "Upload a screen recording",
            type=["mov", "mp4", "m4v", "avi", "mkv"],
            help="Upload a screen recording to analyze"
        )
        
        if uploaded_file:
            # Show file info
            st.info(f"📎 {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.1f} MB)")
            
            # Save uploaded file temporarily
            temp_dir = Path(tempfile.mkdtemp())
            video_path = temp_dir / uploaded_file.name
            
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("🔍 Analyze Video", type="primary", disabled=st.session_state.analyzing):
                analyze_video(video_path)
    
    st.divider()
    
    # Recent recordings
    st.subheader("📁 Recent Recordings")
    
    recordings_dir = Path("./recordings")
    if recordings_dir.exists():
        video_files = list(recordings_dir.glob("*.mov")) + list(recordings_dir.glob("*.mp4"))
        video_files = sorted(video_files, key=lambda p: p.stat().st_mtime, reverse=True)[:10]
        
        if video_files:
            for video_path in video_files:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.text(f"🎬 {video_path.name}")
                with col2:
                    size_mb = video_path.stat().st_size / 1024 / 1024
                    st.text(f"{size_mb:.1f} MB")
                with col3:
                    if st.button("Analyze", key=f"analyze_{video_path.name}", disabled=st.session_state.analyzing):
                        analyze_video(video_path)
        else:
            st.info("No recordings found. Record a video or upload one.")
    else:
        st.info("No recordings directory. Run `python main.py record` to create your first recording.")


def analyze_video(video_path: Path):
    """Analyze a video file and extract workflow."""
    st.session_state.analyzing = True
    
    progress = st.progress(0, "Starting analysis...")
    
    try:
        # Create output directories
        workflows_dir = Path("./workflows")
        workflows_dir.mkdir(exist_ok=True)
        
        processed_dir = Path("./processed") / video_path.stem
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        progress.progress(10, "Processing video (extracting frames)...")
        
        # Extract workflow
        extractor = WorkflowExtractor(model=st.session_state.config.model)
        
        progress.progress(30, "Transcribing audio...")
        
        workflow = extractor.extract_from_video(
            video_path=video_path,
            output_dir=processed_dir,
            max_frames=30,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        
        progress.progress(70, "Detecting parameters...")
        
        # Detect additional parameters
        detector = ParameterDetector(model=st.session_state.config.model)
        suggested_params = detector.detect_parameters(workflow)
        
        # Merge suggested parameters
        for param in suggested_params:
            if param.name not in [p.name for p in workflow.parameters]:
                workflow.parameters.append(param)
        
        progress.progress(90, "Saving workflow...")
        
        # Save both .md and .json files
        output_base = workflows_dir / workflow.id
        md_path, json_path = workflow.save_both(output_base)
        
        progress.progress(100, "Complete!")
        
        st.session_state.current_workflow = workflow
        
        st.success(f"✅ Workflow extracted: **{workflow.name}**")
        st.info(f"📄 Saved to:\n- {md_path}\n- {json_path}")
        
        # Show preview
        with st.expander("Preview Workflow", expanded=True):
            st.markdown(f"### {workflow.name}")
            st.markdown(workflow.description)
            
            if workflow.parameters:
                st.markdown("**Parameters:**")
                for param in workflow.parameters:
                    st.markdown(f"- `{param.name}`: {param.description} (default: `{param.default_value}`)")
        
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        import traceback
        st.code(traceback.format_exc())
    
    finally:
        st.session_state.analyzing = False


def render_review_tab():
    """Render the workflow review tab."""
    st.header("📋 Review Workflows")
    
    # Load existing workflows
    workflows_dir = Path("./workflows")
    workflow_files = get_workflow_files(workflows_dir)
    
    if not workflow_files and not st.session_state.current_workflow:
        st.info("No workflows available. Analyze a video recording first!")
        return
    
    # Workflow selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        workflow_options = []
        if st.session_state.current_workflow:
            workflow_options.append("Current (unsaved)")
        workflow_options.extend([f.stem for f in workflow_files])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_options = []
        for opt in workflow_options:
            if opt not in seen:
                seen.add(opt)
                unique_options.append(opt)
        
        selected = st.selectbox("Select Workflow", unique_options)
    
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Load selected workflow
    workflow = None
    workflow_path = None
    
    if selected == "Current (unsaved)":
        workflow = st.session_state.current_workflow
    else:
        # Find the file (prefer .md over .json)
        for f in workflow_files:
            if f.stem == selected:
                workflow_path = f
                break
        
        if workflow_path:
            workflow = Workflow.load(workflow_path)
            st.session_state.current_workflow = workflow
    
    if not workflow:
        st.error("Could not load workflow")
        return
    
    # Workflow details
    st.subheader(workflow.name)
    st.markdown(workflow.description)
    
    # Show both file formats if they exist
    if workflow_path:
        md_path = workflow_path.with_suffix(".md")
        json_path = workflow_path.with_suffix(".json")
        
        col1, col2 = st.columns(2)
        with col1:
            if md_path.exists():
                st.caption(f"📄 {md_path.name}")
        with col2:
            if json_path.exists():
                st.caption(f"📄 {json_path.name}")
    
    # Parameters
    st.subheader("Parameters")
    
    if workflow.parameters:
        for param in workflow.parameters:
            st.markdown(f"""
            <div class="param-card">
                <strong>{param.name}</strong> ({param.param_type.value})<br>
                {param.description}<br>
                <em>Default: {param.default_value}</em>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No parameters defined for this workflow.")
    
    # Instructions (markdown content)
    st.subheader("Instructions")
    
    with st.expander("View/Edit Instructions", expanded=True):
        # Show the markdown instructions
        edited_instructions = st.text_area(
            "Workflow Instructions (Markdown)",
            value=workflow.instructions,
            height=400,
            help="Edit the workflow instructions here. Use markdown formatting.",
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📖 Preview"):
                st.markdown("---")
                st.markdown(edited_instructions)
        
        with col2:
            if st.button("💾 Save Changes"):
                workflow.instructions = edited_instructions
                
                # Save to both formats
                if workflow_path:
                    output_base = workflow_path.with_suffix("")
                else:
                    output_base = workflows_dir / workflow.id
                
                workflow.save_both(output_base)
                st.success("Workflow saved to both .md and .json!")
                st.rerun()
    
    # Add/Edit parameters
    st.subheader("Edit Parameters")
    
    with st.form("edit_parameters"):
        new_param_name = st.text_input("Parameter Name")
        new_param_desc = st.text_input("Description")
        new_param_default = st.text_input("Default Value")
        new_param_type = st.selectbox("Type", ["string", "number", "boolean", "selection"])
        
        if st.form_submit_button("Add Parameter"):
            workflow.parameters.append(Parameter(
                name=new_param_name,
                param_type=ParameterType(new_param_type),
                description=new_param_desc,
                default_value=new_param_default,
                required=True,
            ))
            
            # Save to both formats
            if workflow_path:
                output_base = workflow_path.with_suffix("")
            else:
                output_base = workflows_dir / workflow.id
            
            workflow.save_both(output_base)
            st.success("Parameter added!")
            st.rerun()


def render_execute_tab():
    """Render the workflow execution tab."""
    st.header("▶️ Execute Workflow")
    
    # Load workflows
    workflows_dir = Path("./workflows")
    workflow_files = get_workflow_files(workflows_dir)
    
    if not workflow_files:
        st.info("No workflows available. Analyze a video recording first!")
        return
    
    # Workflow selector (prefer .md files, deduplicate)
    unique_workflows = {}
    for f in workflow_files:
        stem = f.stem
        if stem not in unique_workflows or f.suffix == ".md":
            unique_workflows[stem] = f
    
    workflow_options = sorted(unique_workflows.keys())
    
    selected = st.selectbox(
        "Select Workflow",
        workflow_options,
        key="exec_workflow_select"
    )
    
    workflow_path = unique_workflows.get(selected)
    
    if not workflow_path:
        st.error("Workflow not found")
        return
    
    workflow = Workflow.load(workflow_path)
    
    st.markdown(f"**{workflow.name}**: {workflow.description}")
    
    # Show preview of instructions
    with st.expander("View Instructions"):
        st.markdown(workflow.instructions)
    
    # Parameter inputs
    st.subheader("Parameters")
    
    param_values = {}
    all_params = workflow.get_all_parameters()
    
    if all_params:
        for param in all_params:
            if param.param_type == ParameterType.BOOLEAN:
                param_values[param.name] = st.checkbox(
                    param.name,
                    value=bool(param.default_value),
                    help=param.description,
                )
            elif param.param_type == ParameterType.NUMBER:
                param_values[param.name] = st.number_input(
                    param.name,
                    value=float(param.default_value or 0),
                    help=param.description,
                )
            elif param.param_type == ParameterType.SELECTION and param.options:
                param_values[param.name] = st.selectbox(
                    param.name,
                    options=param.options,
                    index=param.options.index(param.default_value) if param.default_value in param.options else 0,
                    help=param.description,
                )
            else:
                param_values[param.name] = st.text_input(
                    param.name,
                    value=str(param.default_value or ""),
                    help=param.description,
                )
    else:
        st.info("This workflow has no parameters.")
    
    # Execution controls
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶️ Execute Workflow", type="primary", use_container_width=True, disabled=st.session_state.execution_running):
            execute_workflow(workflow, param_values)
    
    with col2:
        if st.button("🛑 Stop Execution", use_container_width=True, disabled=not st.session_state.execution_running):
            st.session_state.execution_running = False
            st.warning("Execution stopped by user")
    
    # Execution output
    st.subheader("Execution Log")
    
    log_container = st.container()
    
    with log_container:
        for msg in st.session_state.execution_messages:
            if msg["type"] == "text":
                st.markdown(msg["content"])
            elif msg["type"] == "tool":
                with st.expander(f"🔧 Tool: {msg.get('tool', 'computer')}", expanded=False):
                    if msg.get("output"):
                        st.text(msg["output"])
                    if msg.get("image"):
                        st.image(
                            base64.b64decode(msg["image"]),
                            caption="Screenshot",
                            use_container_width=True
                        )
                    if msg.get("error"):
                        st.error(msg["error"])
            elif msg["type"] == "error":
                st.error(msg["content"])
            elif msg["type"] == "success":
                st.success(msg["content"])


def execute_workflow(workflow: Workflow, parameters: dict[str, Any]):
    """Execute a workflow with the given parameters."""
    st.session_state.execution_running = True
    st.session_state.execution_messages = []
    
    def output_callback(content: BetaContentBlockParam):
        if isinstance(content, dict):
            if content.get("type") == "text":
                st.session_state.execution_messages.append({
                    "type": "text",
                    "content": content.get("text", ""),
                })
    
    def tool_callback(result: ToolResult, tool_id: str):
        st.session_state.execution_messages.append({
            "type": "tool",
            "tool_id": tool_id,
            "output": result.output,
            "error": result.error,
            "image": result.base64_image,
        })
    
    try:
        runner = WorkflowRunner(
            model=st.session_state.config.model,
            max_iterations=50,
        )
        
        result = runner.run_sync(
            workflow=workflow,
            parameters=parameters,
            output_callback=output_callback,
            tool_output_callback=tool_callback,
        )
        
        if result.success:
            st.session_state.execution_messages.append({
                "type": "success",
                "content": "Workflow completed successfully!",
            })
        else:
            st.session_state.execution_messages.append({
                "type": "error",
                "content": f"Workflow failed: {result.error}",
            })
        
    except Exception as e:
        st.session_state.execution_messages.append({
            "type": "error",
            "content": f"Execution error: {e}",
        })
    
    finally:
        st.session_state.execution_running = False
        st.rerun()


def main():
    """Main application entry point."""
    setup_state()
    
    st.title("🤖 Workflow Automation")
    st.markdown("*Learn and automate computer tasks from video demonstrations*")
    
    render_sidebar()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🎬 Record & Analyze", "📋 Review", "▶️ Execute"])
    
    with tab1:
        render_record_tab()
    
    with tab2:
        render_review_tab()
    
    with tab3:
        render_execute_tab()


if __name__ == "__main__":
    main()
