#!/usr/bin/env python3
"""
Workflow Automation CLI

A system that learns and automates computer workflows from demonstrations.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

from config import ModelConfig
from utils.logger import WorkflowLogger, set_logger
from utils.tracking import CostTracker, Timer
from utils.llm import LLMClient

load_dotenv()

# Create Typer app
app = typer.Typer(
    name="workflow-automation",
    help="Learn and automate computer tasks from demonstrations",
    rich_markup_mode="rich",
)

console = Console()


@app.command()
def record(
    output: Annotated[
        Optional[Path],
        typer.Option("-o", "--output", help="Output directory for recording"),
    ] = None,
    audio: Annotated[
        bool,
        typer.Option("--audio/--no-audio", help="Enable/disable audio recording"),
    ] = True,
    clicks: Annotated[
        bool,
        typer.Option("--clicks/--no-clicks", help="Show/hide mouse clicks in recording"),
    ] = True,
) -> None:
    """Record a workflow demonstration (screen + audio)."""
    from datetime import datetime
    from recorder.session import record_video
    
    recordings_dir = Path(output or "./recordings")
    recordings_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = recordings_dir / f"recording_{timestamp}.mov"
    
    console.print(f"[bold]Starting screen recording...[/bold]")
    console.print(f"Output: [cyan]{video_path}[/cyan]")
    console.print(f"Audio: [{'green' if audio else 'red'}]{'enabled' if audio else 'disabled'}[/]")
    console.print(f"Show clicks: [{'green' if clicks else 'red'}]{'enabled' if clicks else 'disabled'}[/]")
    console.print()
    console.print("[red bold]🔴 Recording will start immediately.[/red bold]")
    console.print("Press [bold]Ctrl+C[/bold] to stop recording.")
    console.print()
    
    try:
        result_path = record_video(
            output_path=video_path,
            capture_audio=audio,
            capture_clicks=clicks,
        )
        console.print(f"\n[green]✓[/green] Recording saved: [cyan]{result_path}[/cyan]")
        console.print(f"\nTo analyze this recording, run:")
        console.print(f"  [dim]python main.py analyze {result_path}[/dim]")
        
    except KeyboardInterrupt:
        console.print("\n\nRecording stopped.")
        if video_path.exists():
            console.print(f"[green]✓[/green] Recording saved: [cyan]{video_path}[/cyan]")


@app.command()
def analyze(
    video: Annotated[
        Path,
        typer.Argument(help="Path to video file (.mov, .mp4, etc.)"),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("-o", "--output", help="Output path for workflow (without extension)"),
    ] = None,
    event_model: Annotated[
        str,
        typer.Option("--event-model", help="Model for event detection (Pass 1) - fast/cheap"),
    ] = "gemini-3-flash-preview",
    synthesis_model: Annotated[
        str,
        typer.Option("--synthesis-model", "-m", help="Model for understanding/synthesis (Pass 2/3)"),
    ] = "claude-sonnet-4-5-20250929",
    max_frames: Annotated[
        int,
        typer.Option("--max-frames", help="Maximum number of frames to analyze"),
    ] = 30,
    skip_params: Annotated[
        bool,
        typer.Option("--skip-params", help="Skip automatic parameter detection"),
    ] = False,
    cost_optimized: Annotated[
        bool,
        typer.Option("--cost-optimized", help="Use cost-optimized model configuration"),
    ] = True,
) -> None:
    """Analyze a video recording and extract workflow.
    
    Uses a multi-model approach for cost optimization:
    - Pass 1 (Event Detection): Uses fast/cheap model (default: gemini-3-flash-preview)
    - Pass 2/3 (Understanding/Synthesis): Uses stronger model (default: claude-sonnet-4-5)
    """
    from analyzer.workflow_extractor import WorkflowExtractor
    from analyzer.parameter_detector import ParameterDetector
    
    if not video.exists():
        console.print(f"[red]✗[/red] Video file not found: {video}")
        raise typer.Exit(1)
    
    # Check for supported formats
    if video.suffix.lower() not in [".mov", ".mp4", ".m4v", ".avi", ".mkv"]:
        console.print(f"[yellow]⚠[/yellow] Unsupported video format: {video.suffix}")
        console.print("Supported formats: .mov, .mp4, .m4v, .avi, .mkv")
    
    # Create model configuration
    if cost_optimized:
        model_config = ModelConfig.cost_optimized()
    else:
        model_config = ModelConfig.all_same(synthesis_model)
    
    # Override with explicit CLI options if provided
    model_config.event_detection = event_model
    model_config.understanding = synthesis_model
    model_config.synthesis = synthesis_model
    model_config.parameter_detection = synthesis_model
    
    # Initialize logger and tracking
    logger = WorkflowLogger("analyze")
    set_logger(logger)
    cost_tracker = CostTracker()
    llm_client = LLMClient(cost_tracker, logger)
    timer = Timer("Analysis")
    
    try:
        timer.start()
        
        logger.header("Workflow Analysis")
        logger.info(f"Video: [cyan]{video}[/cyan]")
        logger.info(f"Event Model (Pass 1): [cyan]{model_config.event_detection}[/cyan]")
        logger.info(f"Synthesis Model (Pass 2/3): [cyan]{model_config.synthesis}[/cyan]")
        
        # Determine output path
        if output:
            output_base = Path(output)
            if output_base.suffix in [".md", ".json"]:
                output_base = output_base.with_suffix("")
        else:
            output_base = Path("./workflows") / video.stem
        
        # Create processed data directory
        processed_dir = Path("./processed") / video.stem
        
        logger.step("Processing video (extracting frames and audio)...")
        
        extractor = WorkflowExtractor(
            model_config=model_config,
            llm_client=llm_client,
            logger=logger,
        )
        
        try:
            workflow = extractor.extract_from_video(
                video_path=video,
                output_dir=processed_dir,
                max_frames=max_frames,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            import traceback
            traceback.print_exc()
            raise typer.Exit(1)
        
        logger.success(f"Workflow extracted: [bold]{workflow.name}[/bold]")
        logger.info(f"Description: {workflow.description}")
        logger.info(f"Parameters: {len(workflow.parameters)}")
        
        # Detect additional parameters if not skipped
        if not skip_params and workflow.instructions:
            logger.step("Detecting additional parameters...")
            detector = ParameterDetector(
                model_config=model_config,
                llm_client=llm_client,
            )
            suggested = detector.detect_parameters(workflow)
            
            for param in suggested:
                if param.name not in [p.name for p in workflow.parameters]:
                    workflow.parameters.append(param)
            
            logger.info(f"Total parameters: {len(workflow.parameters)}")
        
        # Save both .md and .json files
        output_base.parent.mkdir(parents=True, exist_ok=True)
        md_path, json_path = workflow.save_both(output_base)
        
        logger.success(f"Markdown saved: [cyan]{md_path}[/cyan]")
        logger.success(f"JSON saved: [cyan]{json_path}[/cyan]")
        
        timer.stop()
        
        # Print summary
        logger.header("Analysis Summary")
        
        # Cost breakdown by phase
        if cost_tracker.phase_stats:
            logger.table(
                "Cost by Phase",
                ["Phase", "Model", "Calls", "Input", "Output", "Cost"],
                cost_tracker.get_phase_summary(),
            )
            logger.print()
        
        # Cost breakdown by model
        if cost_tracker.model_stats:
            logger.table(
                "Cost by Model",
                ["Model", "Calls", "Input Tokens", "Output Tokens", "Cost"],
                cost_tracker.get_model_summary(),
            )
            logger.print()
        
        # Overall summary
        summary_data = {
            "Status": "[green]Completed[/green]",
            "Duration": timer.elapsed_str,
            **cost_tracker.get_summary(),
            "Log File": str(logger.log_file),
        }
        logger.summary("Analysis Complete", summary_data)
        
        logger.print()
        logger.info(f"To run this workflow:")
        logger.print(f"  [dim]python main.py run {md_path}[/dim]")
        
    finally:
        logger.close()


@app.command()
def run(
    workflow: Annotated[
        Path,
        typer.Argument(help="Path to workflow file (.md or .json)"),
    ],
    params: Annotated[
        Optional[list[str]],
        typer.Option("-p", "--params", help="Parameters as key=value pairs"),
    ] = None,
    params_file: Annotated[
        Optional[Path],
        typer.Option("--params-file", help="JSON file with parameters"),
    ] = None,
    model: Annotated[
        str,
        typer.Option("-m", "--model", help="Claude model to use"),
    ] = "claude-sonnet-4-5-20250929",
    max_iterations: Annotated[
        int,
        typer.Option("--max-iterations", help="Maximum execution iterations"),
    ] = 50,
) -> None:
    """Run a workflow with parameters."""
    from analyzer.schema import Workflow
    from executor.workflow_runner import WorkflowRunner
    
    if not workflow.exists():
        console.print(f"[red]✗[/red] Workflow not found: {workflow}")
        raise typer.Exit(1)
    
    wf = Workflow.load(workflow)
    
    # Parse parameters
    param_dict: dict[str, str] = {}
    if params:
        for param_str in params:
            if "=" in param_str:
                key, value = param_str.split("=", 1)
                param_dict[key] = value
    
    # Load params from file
    if params_file:
        with open(params_file) as f:
            param_dict.update(json.load(f))
    
    # Initialize logger and tracking
    logger = WorkflowLogger("run")
    set_logger(logger)
    cost_tracker = CostTracker()
    timer = Timer("Execution")
    
    try:
        timer.start()
        
        logger.header("Workflow Execution")
        logger.info(f"Workflow: [bold]{wf.name}[/bold]")
        logger.info(f"Model: [cyan]{model}[/cyan]")
        if param_dict:
            logger.info(f"Parameters: {param_dict}")
        
        iteration_count = 0
        
        def output_callback(content: dict) -> None:
            if isinstance(content, dict) and content.get("type") == "text":
                text = content.get("text", "")
                logger.model_response(text)
        
        def tool_callback(result, tool_id: str) -> None:
            if result.error:
                logger.error(f"Tool error: {result.error}")
            elif result.output:
                logger.info(f"Tool output: {result.output}")
            if result.base64_image:
                logger.step("Screenshot captured")
        
        def iteration_callback(iteration: int, total: int) -> None:
            nonlocal iteration_count
            iteration_count = iteration
            logger.iteration(iteration, total)
        
        def api_callback(input_tokens: int, output_tokens: int) -> None:
            cost_tracker.add_usage(input_tokens, output_tokens, model=model)
            logger.api(input_tokens, output_tokens)
        
        runner = WorkflowRunner(
            model=model,
            max_iterations=max_iterations,
            logger=logger,
            cost_tracker=cost_tracker,
        )
        
        result = asyncio.run(runner.run(
            workflow=wf,
            parameters=param_dict,
            output_callback=output_callback,
            tool_output_callback=tool_callback,
            iteration_callback=iteration_callback,
            api_callback=api_callback,
        ))
        
        timer.stop()
        
        # Print summary
        logger.header("Execution Summary")
        
        status = "[green]Completed[/green]" if result.success else f"[red]Failed: {result.error}[/red]"
        summary_data = {
            "Status": status,
            "Duration": timer.elapsed_str,
            "Iterations": str(iteration_count),
            **cost_tracker.get_summary(),
            "Log File": str(logger.log_file),
        }
        logger.summary("Run Complete", summary_data, style="green" if result.success else "red")
        
        if not result.success:
            raise typer.Exit(1)
            
    finally:
        logger.close()


@app.command("list")
def list_workflows(
    dir: Annotated[
        Optional[Path],
        typer.Option("-d", "--dir", help="Workflows directory"),
    ] = None,
) -> None:
    """List available workflows."""
    from analyzer.schema import Workflow
    
    workflows_dir = Path(dir or "./workflows")
    
    if not workflows_dir.exists():
        console.print("[yellow]No workflows directory found.[/yellow]")
        return
    
    # Support both .md and .json files
    workflow_files = list(workflows_dir.glob("*.md")) + list(workflows_dir.glob("*.json"))
    
    if not workflow_files:
        console.print("[yellow]No workflows found.[/yellow]")
        return
    
    console.print(f"\n[bold]Workflows in {workflows_dir}:[/bold]\n")
    
    for path in sorted(workflow_files, key=lambda p: p.stem):
        try:
            wf = Workflow.load(path)
            console.print(f"  [cyan]📄 {path.name}[/cyan]")
            console.print(f"     Name: [bold]{wf.name}[/bold]")
            desc = wf.description[:60] + "..." if len(wf.description) > 60 else wf.description
            console.print(f"     Description: {desc}")
            params = ", ".join(p.name for p in wf.parameters) or "none"
            console.print(f"     Parameters: [dim]{params}[/dim]")
            console.print()
        except Exception as e:
            console.print(f"  [red]✗ {path.name}[/red]: Error loading - {e}")


@app.command()
def show(
    workflow: Annotated[
        Path,
        typer.Argument(help="Path to workflow file"),
    ],
) -> None:
    """Show details of a workflow."""
    from analyzer.schema import Workflow
    from rich.markdown import Markdown
    from rich.panel import Panel
    
    if not workflow.exists():
        console.print(f"[red]✗[/red] Workflow not found: {workflow}")
        raise typer.Exit(1)
    
    wf = Workflow.load(workflow)
    
    console.print(f"\n[bold blue]# {wf.name}[/bold blue]")
    console.print(f"\n[dim]ID:[/dim] {wf.id}")
    console.print(f"[dim]Description:[/dim] {wf.description}")
    console.print(f"[dim]Created:[/dim] {wf.created_at}")
    
    if wf.parameters:
        console.print(f"\n[bold]## Parameters ({len(wf.parameters)})[/bold]")
        for param in wf.parameters:
            console.print(f"\n  [cyan]{param.name}[/cyan] ([dim]{param.param_type.value}[/dim])")
            console.print(f"    Description: {param.description}")
            console.print(f"    Default: [green]{param.default_value}[/green]")
            console.print(f"    Required: {'Yes' if param.required else 'No'}")
    
    console.print(f"\n[bold]## Instructions[/bold]\n")
    console.print(Panel(Markdown(wf.instructions), border_style="dim"))


@app.command()
def ui(
    port: Annotated[
        Optional[int],
        typer.Option("--port", help="Port for Streamlit server"),
    ] = None,
) -> None:
    """Launch the Streamlit UI."""
    import subprocess
    
    ui_path = Path(__file__).parent / "ui" / "streamlit_app.py"
    
    cmd = ["streamlit", "run", str(ui_path)]
    
    if port:
        cmd.extend(["--server.port", str(port)])
    
    console.print(f"[bold]Launching UI:[/bold] {' '.join(cmd)}")
    subprocess.run(cmd)


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
