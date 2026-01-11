"""High-level workflow runner that orchestrates execution."""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from anthropic.types.beta import BetaContentBlockParam, BetaMessageParam

from analyzer.schema import Workflow
from .loop import workflow_sampling_loop
from .macos_computer import ToolResult

if TYPE_CHECKING:
    from utils.logger import WorkflowLogger
    from utils.tracking import CostTracker


@dataclass
class ExecutionResult:
    """Result of a workflow execution."""
    
    success: bool
    workflow_id: str
    parameters: dict[str, Any]
    messages: list[BetaMessageParam]
    error: str | None = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str | None = None
    
    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "success": self.success,
            "workflow_id": self.workflow_id,
            "parameters": self.parameters,
            "message_count": len(self.messages),
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
    
    def save(self, path: Path) -> None:
        """Save execution result to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class WorkflowRunner:
    """Runs workflows with parameter substitution."""
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 16384,
        max_iterations: int = 100,
        logger: "WorkflowLogger | None" = None,
        cost_tracker: "CostTracker | None" = None,
    ):
        """Initialize the runner.
        
        Args:
            api_key: Anthropic API key.
            model: Claude model to use.
            max_tokens: Maximum tokens per response.
            max_iterations: Maximum loop iterations.
            logger: Optional WorkflowLogger for structured output.
            cost_tracker: Optional CostTracker for cost accumulation.
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.logger = logger
        self.cost_tracker = cost_tracker
    
    async def run(
        self,
        workflow: Workflow,
        parameters: dict[str, Any],
        output_callback: Callable[[BetaContentBlockParam], None] | None = None,
        tool_output_callback: Callable[[ToolResult, str], None] | None = None,
        iteration_callback: Callable[[int, int], None] | None = None,
        api_callback: Callable[[int, int], None] | None = None,
    ) -> ExecutionResult:
        """Run a workflow with the given parameters.
        
        Args:
            workflow: The workflow to execute.
            parameters: Parameter values to use.
            output_callback: Callback for model outputs.
            tool_output_callback: Callback for tool results.
            iteration_callback: Callback for iteration progress.
            api_callback: Callback for API token usage.
            
        Returns:
            ExecutionResult with success status and messages.
        """
        # Log start if logger available
        if self.logger:
            self.logger.step(f"Starting workflow execution: {workflow.name}")
        
        # Validate parameters
        errors = workflow.validate_parameters(parameters)
        if errors:
            error_msg = f"Parameter validation failed: {', '.join(errors)}"
            if self.logger:
                self.logger.error(error_msg)
            return ExecutionResult(
                success=False,
                workflow_id=workflow.id,
                parameters=parameters,
                messages=[],
                error=error_msg,
                completed_at=datetime.now().isoformat(),
            )
        
        # Fill in defaults for missing optional parameters
        full_params = workflow.fill_defaults(parameters)
        
        try:
            # Pass the markdown instructions directly to the loop
            messages = await workflow_sampling_loop(
                workflow_instructions=workflow.instructions,
                parameters=full_params,
                model=self.model,
                api_key=self.api_key,
                output_callback=output_callback,
                tool_output_callback=tool_output_callback,
                iteration_callback=iteration_callback,
                api_usage_callback=api_callback,
                max_tokens=self.max_tokens,
                max_iterations=self.max_iterations,
                logger=self.logger,
                cost_tracker=self.cost_tracker,
                processed_session_path=workflow.processed_session_path,
            )
            
            if self.logger:
                self.logger.success("Workflow execution completed")
            
            return ExecutionResult(
                success=True,
                workflow_id=workflow.id,
                parameters=full_params,
                messages=messages,
                completed_at=datetime.now().isoformat(),
            )
            
        except Exception as e:
            error_msg = str(e)
            if self.logger:
                self.logger.error(f"Workflow execution failed: {error_msg}")
            return ExecutionResult(
                success=False,
                workflow_id=workflow.id,
                parameters=full_params,
                messages=[],
                error=error_msg,
                completed_at=datetime.now().isoformat(),
            )
    
    def run_sync(
        self,
        workflow: Workflow,
        parameters: dict[str, Any],
        output_callback: Callable[[BetaContentBlockParam], None] | None = None,
        tool_output_callback: Callable[[ToolResult, str], None] | None = None,
        iteration_callback: Callable[[int, int], None] | None = None,
        api_callback: Callable[[int, int], None] | None = None,
    ) -> ExecutionResult:
        """Synchronous wrapper for run()."""
        return asyncio.run(self.run(
            workflow=workflow,
            parameters=parameters,
            output_callback=output_callback,
            tool_output_callback=tool_output_callback,
            iteration_callback=iteration_callback,
            api_callback=api_callback,
        ))
    
    @staticmethod
    def load_workflow(path: Path) -> Workflow:
        """Load a workflow from a file (.md or .json)."""
        return Workflow.load(path)
    
    @staticmethod
    def list_workflows(workflows_dir: Path) -> list[tuple[Path, Workflow]]:
        """List all workflows in a directory."""
        workflows = []
        
        # Support both .md and .json files
        for pattern in ["*.md", "*.json"]:
            for path in workflows_dir.glob(pattern):
                try:
                    workflow = Workflow.load(path)
                    workflows.append((path, workflow))
                except Exception:
                    pass  # Skip invalid files
        
        return workflows
