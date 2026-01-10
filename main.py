#!/usr/bin/env python3
"""
Workflow Automation CLI

A system that learns and automates computer workflows from demonstrations.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()


def cmd_record(args):
    """Start a screen recording session using macOS screencapture."""
    from datetime import datetime
    
    recordings_dir = Path(args.output or "./recordings")
    recordings_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = recordings_dir / f"recording_{timestamp}.mov"
    
    print(f"Starting screen recording...")
    print(f"Output: {video_path}")
    print(f"Audio: {'enabled' if args.audio else 'disabled'}")
    print(f"Show clicks: {'enabled' if args.clicks else 'disabled'}")
    print()
    print("🔴 Recording will start immediately.")
    print("Press Ctrl+C to stop recording.")
    print()
    
    from recorder.session import record_video
    
    try:
        result_path = record_video(
            output_path=video_path,
            capture_audio=args.audio,
            capture_clicks=args.clicks,
        )
        print(f"\n✅ Recording saved: {result_path}")
        print(f"\nTo analyze this recording, run:")
        print(f"  python main.py analyze {result_path}")
        
    except KeyboardInterrupt:
        print("\n\nRecording stopped.")
        if video_path.exists():
            print(f"✅ Recording saved: {video_path}")


def cmd_analyze(args):
    """Analyze a video recording and extract workflow."""
    from analyzer.workflow_extractor import WorkflowExtractor
    from analyzer.parameter_detector import ParameterDetector
    
    video_path = Path(args.video)
    
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Check for supported formats
    if video_path.suffix.lower() not in [".mov", ".mp4", ".m4v", ".avi", ".mkv"]:
        print(f"Warning: Unsupported video format: {video_path.suffix}")
        print("Supported formats: .mov, .mp4, .m4v, .avi, .mkv")
    
    # Set default model based on provider
    if args.use_openai and args.model == "claude-sonnet-4-5-20250929":
        args.model = "gpt-5-mini-2025-08-07"
    
    provider = "OpenAI" if args.use_openai else "Anthropic"
    print(f"Analyzing video: {video_path}")
    print(f"Using provider: {provider}")
    print(f"Using model: {args.model}")
    print()
    
    # Determine output path
    if args.output:
        output_base = Path(args.output)
        # Remove extension if provided
        if output_base.suffix in [".md", ".json"]:
            output_base = output_base.with_suffix("")
    else:
        output_base = Path("./workflows") / video_path.stem
    
    # Create processed data directory
    processed_dir = Path("./processed") / video_path.stem
    
    print(f"Processing video (extracting frames and audio)...")
    
    extractor = WorkflowExtractor(model=args.model, use_openai=args.use_openai)
    
    try:
        workflow = extractor.extract_from_video(
            video_path=video_path,
            output_dir=processed_dir,
            max_frames=args.max_frames,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    except Exception as e:
        print(f"Error during extraction: {e}")
        sys.exit(1)
    
    print(f"\n✅ Workflow extracted: {workflow.name}")
    print(f"   Description: {workflow.description}")
    print(f"   Parameters: {len(workflow.parameters)}")
    
    # Detect additional parameters if not skipped
    if not args.skip_params and workflow.instructions:
        print("\nDetecting additional parameters...")
        detector = ParameterDetector(model=args.model, use_openai=args.use_openai)
        suggested = detector.detect_parameters(workflow)
        
        for param in suggested:
            if param.name not in [p.name for p in workflow.parameters]:
                workflow.parameters.append(param)
        
        print(f"   Total parameters: {len(workflow.parameters)}")
    
    # Save both .md and .json files
    output_base.parent.mkdir(parents=True, exist_ok=True)
    md_path, json_path = workflow.save_both(output_base)
    
    print(f"\n📄 Workflow files saved:")
    print(f"   Markdown: {md_path}")
    print(f"   JSON: {json_path}")
    
    # Show preview of instructions
    print("\n--- Instructions Preview ---")
    preview = workflow.instructions[:500]
    if len(workflow.instructions) > 500:
        preview += "..."
    print(preview)
    
    print(f"\nTo run this workflow:")
    print(f"  python main.py run {md_path}")


def cmd_run(args):
    """Run a workflow with parameters."""
    from analyzer.schema import Workflow
    from executor.workflow_runner import WorkflowRunner
    
    workflow_path = Path(args.workflow)
    
    if not workflow_path.exists():
        print(f"Error: Workflow not found: {workflow_path}")
        sys.exit(1)
    
    workflow = Workflow.load(workflow_path)
    
    # Parse parameters
    params = {}
    if args.params:
        for param_str in args.params:
            if "=" in param_str:
                key, value = param_str.split("=", 1)
                params[key] = value
    
    # Load params from file
    if args.params_file:
        with open(args.params_file) as f:
            params.update(json.load(f))
    
    print(f"Running workflow: {workflow.name}")
    print(f"Parameters: {params}")
    print()
    
    def output_callback(content):
        if isinstance(content, dict) and content.get("type") == "text":
            text = content.get('text', '')
            # Print full text for better visibility
            print(f"\n[Claude]\n{text}\n")
    
    def tool_callback(result, tool_id):
        if result.error:
            print(f"[Tool Error] {result.error}")
        elif result.output:
            print(f"[Tool] {result.output}")
        if result.base64_image:
            print(f"[Screenshot taken]")
    
    runner = WorkflowRunner(
        model=args.model,
        max_iterations=args.max_iterations,
    )
    
    result = asyncio.run(runner.run(
        workflow=workflow,
        parameters=params,
        output_callback=output_callback,
        tool_output_callback=tool_callback,
    ))
    
    if result.success:
        print("\n✅ Workflow completed successfully!")
    else:
        print(f"\n❌ Workflow failed: {result.error}")
        sys.exit(1)


def cmd_list(args):
    """List available workflows."""
    from analyzer.schema import Workflow
    
    workflows_dir = Path(args.dir or "./workflows")
    
    if not workflows_dir.exists():
        print("No workflows directory found.")
        return
    
    # Support both .md and .json files
    workflow_files = list(workflows_dir.glob("*.md")) + list(workflows_dir.glob("*.json"))
    
    if not workflow_files:
        print("No workflows found.")
        return
    
    print(f"Workflows in {workflows_dir}:\n")
    
    for path in sorted(workflow_files, key=lambda p: p.stem):
        try:
            workflow = Workflow.load(path)
            print(f"  📄 {path.name}")
            print(f"     Name: {workflow.name}")
            print(f"     Description: {workflow.description[:60]}{'...' if len(workflow.description) > 60 else ''}")
            print(f"     Parameters: {', '.join(p.name for p in workflow.parameters) or 'none'}")
            print()
        except Exception as e:
            print(f"  ❌ {path.name}: Error loading - {e}")


def cmd_show(args):
    """Show details of a workflow."""
    from analyzer.schema import Workflow
    
    workflow_path = Path(args.workflow)
    
    if not workflow_path.exists():
        print(f"Error: Workflow not found: {workflow_path}")
        sys.exit(1)
    
    workflow = Workflow.load(workflow_path)
    
    print(f"# {workflow.name}")
    print(f"\nID: {workflow.id}")
    print(f"Description: {workflow.description}")
    print(f"Created: {workflow.created_at}")
    
    if workflow.parameters:
        print(f"\n## Parameters ({len(workflow.parameters)})")
        for param in workflow.parameters:
            print(f"\n### {param.name} ({param.param_type.value})")
            print(f"  Description: {param.description}")
            print(f"  Default: {param.default_value}")
            print(f"  Required: {param.required}")
    
    print(f"\n## Instructions\n")
    print(workflow.instructions)


def cmd_ui(args):
    """Launch the Streamlit UI."""
    import subprocess
    
    ui_path = Path(__file__).parent / "ui" / "streamlit_app.py"
    
    cmd = ["streamlit", "run", str(ui_path)]
    
    if args.port:
        cmd.extend(["--server.port", str(args.port)])
    
    print(f"Launching UI: {' '.join(cmd)}")
    subprocess.run(cmd)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Workflow Automation - Learn and automate computer tasks from demonstrations"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Record command
    record_parser = subparsers.add_parser(
        "record",
        help="Record a workflow demonstration (screen + audio)"
    )
    record_parser.add_argument(
        "-o", "--output",
        help="Output directory for recording (default: ./recordings)"
    )
    record_parser.add_argument(
        "--no-audio",
        dest="audio",
        action="store_false",
        help="Disable audio recording"
    )
    record_parser.add_argument(
        "--no-clicks",
        dest="clicks",
        action="store_false",
        help="Don't show mouse clicks in recording"
    )
    record_parser.set_defaults(func=cmd_record, audio=True, clicks=True)
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a video recording and extract workflow"
    )
    analyze_parser.add_argument(
        "video",
        help="Path to video file (.mov, .mp4, etc.)"
    )
    analyze_parser.add_argument(
        "-o", "--output",
        help="Output path for workflow (without extension, saves both .md and .json)"
    )
    analyze_parser.add_argument(
        "-m", "--model",
        default="claude-sonnet-4-5-20250929",
        help="Model to use for analysis (default: claude-sonnet-4-5-20250929, or gpt-5-mini-2025-08-07 if --use-openai is set)"
    )
    analyze_parser.add_argument(
        "--max-frames",
        type=int,
        default=30,
        help="Maximum number of frames to analyze (default: 30)"
    )
    analyze_parser.add_argument(
        "--skip-params",
        action="store_true",
        help="Skip automatic parameter detection"
    )
    analyze_parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use OpenAI API instead of Anthropic (default model: gpt-5-mini-2025-08-07)"
    )
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run a workflow")
    run_parser.add_argument("workflow", help="Path to workflow file (.md or .json)")
    run_parser.add_argument("-p", "--params", nargs="+", help="Parameters as key=value pairs")
    run_parser.add_argument("--params-file", help="JSON file with parameters")
    run_parser.add_argument("-m", "--model", default="claude-sonnet-4-5-20250929", help="Claude model to use")
    run_parser.add_argument("--max-iterations", type=int, default=50, help="Maximum execution iterations")
    run_parser.set_defaults(func=cmd_run)
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available workflows")
    list_parser.add_argument("-d", "--dir", help="Workflows directory")
    list_parser.set_defaults(func=cmd_list)
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show details of a workflow")
    show_parser.add_argument("workflow", help="Path to workflow file")
    show_parser.set_defaults(func=cmd_show)
    
    # UI command
    ui_parser = subparsers.add_parser("ui", help="Launch the Streamlit UI")
    ui_parser.add_argument("--port", type=int, help="Port for Streamlit server")
    ui_parser.set_defaults(func=cmd_ui)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
