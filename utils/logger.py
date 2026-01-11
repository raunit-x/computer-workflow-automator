"""Logging utility with Rich console output and file logging."""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.text import Text
from rich.theme import Theme


# Custom theme for consistent styling
THEME = Theme({
    "info": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "red bold",
    "step": "blue",
    "api": "magenta",
    "dim": "dim",
})


class WorkflowLogger:
    """Logger that outputs to Rich console and optionally to a file."""
    
    def __init__(
        self,
        command: str,
        logs_dir: Path | str = "./logs",
        console: Console | None = None,
    ):
        """Initialize the logger.
        
        Args:
            command: The command name (e.g., 'analyze', 'run') for log filename.
            logs_dir: Directory to store log files.
            console: Optional Rich console instance.
        """
        self.command = command
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"{command}_{timestamp}.log"
        self._file_handle = open(self.log_file, "w", encoding="utf-8")
        
        # Rich console for terminal output
        self.console = console or Console(theme=THEME)
        
        # Track if we're in a progress context
        self._progress: Progress | None = None
    
    def _write_to_file(self, level: str, message: str) -> None:
        """Write a log entry to the file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._file_handle.write(f"[{timestamp}] {level}: {message}\n")
        self._file_handle.flush()
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info message."""
        self.console.print(f"[info]ℹ[/info] {message}", **kwargs)
        self._write_to_file("INFO", message)
    
    def success(self, message: str, **kwargs: Any) -> None:
        """Log a success message."""
        self.console.print(f"[success]✓[/success] {message}", **kwargs)
        self._write_to_file("SUCCESS", message)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message."""
        self.console.print(f"[warning]⚠[/warning] {message}", **kwargs)
        self._write_to_file("WARNING", message)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error message."""
        self.console.print(f"[error]✗[/error] {message}", **kwargs)
        self._write_to_file("ERROR", message)
    
    def step(self, message: str, **kwargs: Any) -> None:
        """Log a step/progress message."""
        self.console.print(f"[step]→[/step] {message}", **kwargs)
        self._write_to_file("STEP", message)
    
    def api(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        console: bool = True,
        tqdm_bar: Any = None,
        cumulative_in: int | None = None,
        cumulative_out: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Log API token usage.
        
        Args:
            input_tokens: Input tokens for this call.
            output_tokens: Output tokens for this call.
            console: Whether to print to console (set False when using tqdm).
            tqdm_bar: Optional tqdm progress bar to update with cumulative tokens.
            cumulative_in: Cumulative input tokens (for tqdm display).
            cumulative_out: Cumulative output tokens (for tqdm display).
            **kwargs: Additional kwargs for console.print.
        """
        message = f"Tokens: {input_tokens:,} in, {output_tokens:,} out"
        
        if tqdm_bar is not None and cumulative_in is not None:
            # Update tqdm postfix with cumulative tokens
            tqdm_bar.set_postfix_str(f"⚡ {cumulative_in:,} in, {cumulative_out:,} out")
        elif console:
            self.console.print(f"[api]⚡[/api] {message}", **kwargs)
        
        self._write_to_file("API", message)
    
    def iteration(self, current: int, total: int, **kwargs: Any) -> None:
        """Log iteration progress."""
        message = f"Iteration {current}/{total}"
        self.console.print(f"[dim]│[/dim] {message}", **kwargs)
        self._write_to_file("ITER", message)
    
    def tool(self, tool_name: str, action: str, **kwargs: Any) -> None:
        """Log a tool call."""
        message = f"Tool: {tool_name} → {action}"
        self.console.print(f"[dim]│[/dim] [cyan]{message}[/cyan]", **kwargs)
        self._write_to_file("TOOL", message)
    
    def model_response(self, text: str, **kwargs: Any) -> None:
        """Log model response text."""
        # Truncate for console if very long
        display_text = text[:500] + "..." if len(text) > 500 else text
        self.console.print(Panel(display_text, title="[bold]Claude[/bold]", border_style="blue"))
        self._write_to_file("MODEL", text)
    
    def header(self, title: str, **kwargs: Any) -> None:
        """Print a section header."""
        self.console.print()
        self.console.rule(f"[bold]{title}[/bold]", **kwargs)
        self.console.print()
        self._write_to_file("HEADER", title)
    
    def summary(
        self,
        title: str,
        data: dict[str, str],
        style: str = "green",
    ) -> None:
        """Print a summary panel with key-value data."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold")
        table.add_column("Value")
        
        for key, value in data.items():
            table.add_row(key, value)
        
        panel = Panel(table, title=f"[bold]{title}[/bold]", border_style=style)
        self.console.print(panel)
        
        # Write plain text version to file
        self._write_to_file("SUMMARY", title)
        for key, value in data.items():
            self._write_to_file("SUMMARY", f"  {key}: {value}")
    
    def table(
        self,
        title: str,
        columns: list[str],
        rows: list[list[str]],
        **kwargs: Any,
    ) -> None:
        """Print a table."""
        table = Table(title=title, **kwargs)
        for col in columns:
            table.add_column(col)
        for row in rows:
            table.add_row(*row)
        
        self.console.print(table)
        
        # Write to file
        self._write_to_file("TABLE", title)
        for row in rows:
            self._write_to_file("TABLE", "  " + " | ".join(row))
    
    def progress(
        self,
        description: str = "Processing",
        total: int | None = None,
    ) -> Progress:
        """Create a progress bar context.
        
        Usage:
            with logger.progress("Analyzing", total=100) as progress:
                task = progress.add_task("Frames", total=100)
                for i in range(100):
                    progress.update(task, advance=1)
        """
        if total is not None:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console,
            )
        else:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            )
        return self._progress
    
    def print(self, *args: Any, **kwargs: Any) -> None:
        """Direct print to console (for compatibility)."""
        self.console.print(*args, **kwargs)
        # Try to extract text for file logging
        if args:
            text = " ".join(str(a) for a in args)
            self._write_to_file("PRINT", text)
    
    def close(self) -> None:
        """Close the log file."""
        if self._file_handle:
            self._file_handle.close()
    
    def __enter__(self) -> "WorkflowLogger":
        return self
    
    def __exit__(self, *args: Any) -> None:
        self.close()


# Global logger instance (set by commands)
_current_logger: WorkflowLogger | None = None


def get_logger() -> WorkflowLogger:
    """Get the current logger instance."""
    if _current_logger is None:
        # Return a default logger if none is set
        return WorkflowLogger("default")
    return _current_logger


def set_logger(logger: WorkflowLogger) -> None:
    """Set the current global logger."""
    global _current_logger
    _current_logger = logger

