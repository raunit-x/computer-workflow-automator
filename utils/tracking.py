"""Cost and time tracking utilities."""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# Model pricing per million tokens (input, output)
# From: https://platform.claude.com/docs/en/about-claude/pricing
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Claude Opus 4.5
    "claude-opus-4-5-20250929": (5.0, 25.0),
    "claude-opus-4-5": (5.0, 25.0),
    # Claude Opus 4.1
    "claude-opus-4-1-20250414": (15.0, 75.0),
    "claude-opus-4-1": (15.0, 75.0),
    # Claude Opus 4
    "claude-opus-4-20250514": (15.0, 75.0),
    "claude-opus-4": (15.0, 75.0),
    # Claude Sonnet 4.5
    "claude-sonnet-4-5-20250929": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    # Claude Sonnet 4
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-sonnet-4": (3.0, 15.0),
    # Claude Haiku 4.5
    "claude-haiku-4-5-20250929": (1.0, 5.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

# OpenAI model pricing (approximate)
OPENAI_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5-mini-2025-08-07": (0.15, 0.60),
    "gpt-5-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-2024-08-06": (2.50, 10.0),
    "gpt-4-turbo": (10.0, 30.0),
}

# Gemini model pricing per million tokens (input, output)
# From: https://ai.google.dev/gemini-api/docs/pricing
GEMINI_PRICING: dict[str, tuple[float, float]] = {
    "gemini-3-flash-preview": (0.5, 3.0),
    # Gemini 2.0 Flash
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-flash-exp": (0.10, 0.40),
}

# Default pricing (Claude Sonnet 4.5)
DEFAULT_PRICING = (3.0, 15.0)


def get_model_pricing(model: str) -> tuple[float, float]:
    """Get pricing for a model. Returns (input_price_per_mtok, output_price_per_mtok)."""
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    if model in OPENAI_PRICING:
        return OPENAI_PRICING[model]
    if model in GEMINI_PRICING:
        return GEMINI_PRICING[model]
    return DEFAULT_PRICING


def detect_provider(model: str) -> str:
    """Detect the provider based on model name.
    
    Returns: "anthropic", "openai", or "gemini"
    """
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
        return "openai"
    if model.startswith("gemini"):
        return "gemini"
    # Default to anthropic
    return "anthropic"


@dataclass
class CostTracker:
    """Tracks API costs and token usage across multiple calls.
    
    Supports tracking usage across multiple models with per-model cost calculation.
    """
    
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    api_calls: int = 0
    total_cost_dollars: float = 0.0
    
    # Per-phase tracking for analyze command
    phase_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    # Per-model tracking
    model_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    def add_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        phase: str | None = None,
    ) -> float:
        """Record token usage from an API call.
        
        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            model: The model used for this API call.
            phase: Optional phase name (e.g., 'pass1', 'pass2', 'pass3').
            
        Returns:
            The cost of this API call in dollars.
        """
        # Calculate cost for this call
        input_price, output_price = get_model_pricing(model)
        call_cost = (
            (input_tokens / 1_000_000) * input_price +
            (output_tokens / 1_000_000) * output_price
        )
        
        # Update totals
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.api_calls += 1
        self.total_cost_dollars += call_cost
        
        # Track per-model stats
        if model not in self.model_stats:
            self.model_stats[model] = {"input": 0, "output": 0, "calls": 0, "cost": 0.0}
        self.model_stats[model]["input"] += input_tokens
        self.model_stats[model]["output"] += output_tokens
        self.model_stats[model]["calls"] += 1
        self.model_stats[model]["cost"] += call_cost
        
        # Track per-phase stats
        if phase:
            if phase not in self.phase_stats:
                self.phase_stats[phase] = {"input": 0, "output": 0, "calls": 0, "cost": 0.0, "model": model}
            self.phase_stats[phase]["input"] += input_tokens
            self.phase_stats[phase]["output"] += output_tokens
            self.phase_stats[phase]["calls"] += 1
            self.phase_stats[phase]["cost"] += call_cost
        
        return call_cost
    
    @property
    def total_cost(self) -> float:
        """Get total cost in dollars."""
        return self.total_cost_dollars
    
    def get_summary(self) -> dict[str, str]:
        """Get a summary dictionary for display."""
        # List unique models used
        models_used = list(self.model_stats.keys())
        models_str = ", ".join(models_used) if models_used else "None"
        
        return {
            "Models Used": models_str,
            "API Calls": str(self.api_calls),
            "Input Tokens": f"{self.total_input_tokens:,}",
            "Output Tokens": f"{self.total_output_tokens:,}",
            "Total Cost": f"${self.total_cost:.4f}",
        }
    
    def get_model_summary(self) -> list[list[str]]:
        """Get per-model breakdown for table display."""
        rows = []
        for model, stats in self.model_stats.items():
            input_tokens = stats["input"]
            output_tokens = stats["output"]
            calls = stats["calls"]
            cost = stats["cost"]
            
            rows.append([
                model,
                str(calls),
                f"{input_tokens:,}",
                f"{output_tokens:,}",
                f"${cost:.4f}",
            ])
        return rows
    
    def get_phase_summary(self) -> list[list[str]]:
        """Get per-phase breakdown for table display."""
        rows = []
        for phase, stats in self.phase_stats.items():
            input_tokens = stats["input"]
            output_tokens = stats["output"]
            calls = stats["calls"]
            cost = stats["cost"]
            model = stats.get("model", "unknown")
            
            rows.append([
                phase,
                model,
                str(calls),
                f"{input_tokens:,}",
                f"{output_tokens:,}",
                f"${cost:.4f}",
            ])
        return rows


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time: float | None = None
        self.end_time: float | None = None
    
    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args: Any) -> None:
        self.end_time = time.time()
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def elapsed_str(self) -> str:
        """Get elapsed time as a formatted string."""
        seconds = self.elapsed
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        secs = seconds % 60
        if minutes < 60:
            return f"{minutes}m {secs:.0f}s"
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours}h {mins}m {secs:.0f}s"
    
    def start(self) -> None:
        """Manually start the timer."""
        self.start_time = time.time()
    
    def stop(self) -> float:
        """Manually stop the timer and return elapsed time."""
        self.end_time = time.time()
        return self.elapsed

