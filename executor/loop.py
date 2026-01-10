"""Agentic sampling loop for workflow execution."""

import json
from collections.abc import Callable
from datetime import datetime
from typing import Any, cast

import httpx
from anthropic import Anthropic, APIError, APIStatusError, APIResponseValidationError
from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlock,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
    BetaToolUseBlock,
    BetaToolUseBlockParam,
    BetaToolUnionParam,
)

from .macos_computer import MacOSComputerTool, ToolResult, ToolError


WORKFLOW_SYSTEM_PROMPT = """You are an AI agent executing a learned workflow on macOS.

# Current Date
{current_date}

# Parameters for This Execution
{parameters_section}

# Workflow Instructions
{workflow_instructions}

---

# Execution Guidelines

You have access to a `computer` tool that can:
- `screenshot`: Capture the current screen
- `click`: Click at coordinates or current position
- `double_click`: Double-click
- `right_click`: Right-click
- `type`: Type text
- `key`: Press key combinations (e.g., "cmd+c", "enter")
- `scroll`: Scroll up/down/left/right
- `wait`: Wait for a duration
- `mouse_move`: Move mouse to coordinates
- `cursor_position`: Get current cursor position

## How to Execute

1. **Start** by taking a screenshot to see the current screen state
2. **Read** the workflow instructions above
3. **Execute** each step, substituting `{{parameter_name}}` with the actual values
4. **Verify** each action succeeded by checking the screenshot
5. **Adapt** if the UI looks different - use your judgment to find elements
6. **Report** progress and indicate when complete

## Verification Protocol

After each step, take a screenshot and carefully evaluate if you have achieved the right outcome. Explicitly show your thinking: "I have evaluated step X..." If not correct, try again. Only when you confirm a step was executed correctly should you move on to the next one.

## UI Interaction Tips

- If dropdowns or scrollbars are tricky to manipulate with mouse, use keyboard shortcuts instead
- Wait for UI to fully load before interacting with elements
- For text fields, click to focus first, then type
- Use Tab key to navigate between form fields when appropriate

## Important Rules

- Always take a screenshot after major actions to verify success
- If an action fails, try alternative approaches before giving up
- Use keyboard shortcuts when they're more reliable than clicking
- Wait for UI to settle after actions that trigger loading
- Substitute ALL parameter placeholders with actual values

## When You're Done

Stop calling tools and provide a summary of what was accomplished.
"""


def _format_parameters_section(parameters: dict[str, Any]) -> str:
    """Format parameters for display in the system prompt."""
    if not parameters:
        return "No parameters provided - use default values from the workflow."
    
    lines = ["The following parameter values should be used:\n"]
    for name, value in parameters.items():
        lines.append(f"- **{name}**: `{value}`")
    
    lines.append("\nReplace any `{" + "parameter_name}` placeholders with these values.")
    return "\n".join(lines)


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

# Default pricing (Claude Sonnet 4.5)
DEFAULT_PRICING = (3.0, 15.0)


def _get_model_pricing(model: str) -> tuple[float, float]:
    """Get pricing for a model. Returns (input_price_per_mtok, output_price_per_mtok)."""
    return MODEL_PRICING.get(model, DEFAULT_PRICING)


def _print_cost_summary(total_input_tokens: int, total_output_tokens: int, model: str) -> None:
    """Print a summary of token usage and estimated cost."""
    input_price, output_price = _get_model_pricing(model)
    
    input_cost = (total_input_tokens / 1_000_000) * input_price
    output_cost = (total_output_tokens / 1_000_000) * output_price
    total_cost = input_cost + output_cost
    
    print(f"\n{'='*50}")
    print(f"[Cost Summary] Model: {model}")
    print(f"  Pricing: ${input_price}/MTok in, ${output_price}/MTok out")
    print(f"  Input tokens:  {total_input_tokens:,} (${input_cost:.4f})")
    print(f"  Output tokens: {total_output_tokens:,} (${output_cost:.4f})")
    print(f"  Total cost:    ${total_cost:.4f}")
    print(f"{'='*50}")


async def workflow_sampling_loop(
    *,
    workflow_instructions: str,
    parameters: dict[str, Any],
    model: str = "claude-sonnet-4-5-20250929",
    api_key: str | None = None,
    output_callback: Callable[[BetaContentBlockParam], None] | None = None,
    tool_output_callback: Callable[[ToolResult, str], None] | None = None,
    api_response_callback: Callable[
        [httpx.Request, httpx.Response | object | None, Exception | None], None
    ] | None = None,
    max_tokens: int = 16384,
    max_iterations: int = 50,
) -> list[BetaMessageParam]:
    """
    Agentic sampling loop for executing a learned workflow.
    
    Args:
        workflow_instructions: Markdown instructions for the workflow.
        parameters: Parameter values to use for execution.
        model: Claude model to use.
        api_key: Anthropic API key.
        output_callback: Callback for model outputs.
        tool_output_callback: Callback for tool results.
        api_response_callback: Callback for API responses.
        max_tokens: Maximum tokens for responses.
        max_iterations: Maximum number of loop iterations.
        
    Returns:
        List of conversation messages.
    """
    client = Anthropic(api_key=api_key) if api_key else Anthropic()
    computer_tool = MacOSComputerTool()
    
    # Track token usage for cost calculation
    total_input_tokens = 0
    total_output_tokens = 0
    
    # Build system prompt with workflow instructions
    system_prompt = WORKFLOW_SYSTEM_PROMPT.format(
        current_date=datetime.today().strftime("%A, %B %d, %Y"),
        parameters_section=_format_parameters_section(parameters),
        workflow_instructions=workflow_instructions,
    )
    
    system = BetaTextBlockParam(type="text", text=system_prompt)
    
    # Initial message to start execution
    messages: list[BetaMessageParam] = [
        {
            "role": "user",
            "content": [
                BetaTextBlockParam(
                    type="text",
                    text="Please execute the workflow now. Start by taking a screenshot to see the current screen state, then follow the instructions."
                )
            ],
        }
    ]
    
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        print(f"[Loop] Iteration {iteration}/{max_iterations}")
        
        try:
            print(f"[Loop] Calling API with model: {model}")
            raw_response = client.beta.messages.with_raw_response.create(
                max_tokens=max_tokens,
                messages=messages,
                model=model,
                system=[system],
                tools=[cast(BetaToolUnionParam, computer_tool.to_params())],
                betas=["computer-use-2025-01-24"],
            )
            print(f"[Loop] API call succeeded")
        except (APIStatusError, APIResponseValidationError) as e:
            print(f"[Loop] API Error: {e}")
            if api_response_callback:
                api_response_callback(e.request, e.response, e)
            raise  # Re-raise so WorkflowRunner knows it failed
        except APIError as e:
            print(f"[Loop] API Error: {e}")
            if api_response_callback:
                api_response_callback(e.request, e.body, e)
            raise  # Re-raise so WorkflowRunner knows it failed
        
        if api_response_callback:
            api_response_callback(
                raw_response.http_response.request,
                raw_response.http_response,
                None,
            )
        
        response = raw_response.parse()
        response_params = _response_to_params(response)
        
        # Track token usage
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        
        print(f"[Loop] Response has {len(response_params)} content blocks")
        print(f"[Loop] Stop reason: {response.stop_reason}")
        print(f"[Loop] Tokens: {input_tokens:,} in, {output_tokens:,} out")
        
        messages.append({
            "role": "assistant",
            "content": response_params,
        })
        
        # Process tool calls
        tool_result_content: list[BetaToolResultBlockParam] = []
        
        for content_block in response_params:
            if output_callback:
                output_callback(content_block)
            
            block_type = content_block.get("type") if isinstance(content_block, dict) else "unknown"
            print(f"[Loop] Processing block: {block_type}")
            
            if isinstance(content_block, dict) and content_block.get("type") == "tool_use":
                tool_use_block = cast(BetaToolUseBlockParam, content_block)
                tool_name = tool_use_block.get("name", "unknown")
                tool_input = tool_use_block.get("input", {})
                action = tool_input.get("action", "unknown") if isinstance(tool_input, dict) else "unknown"
                print(f"[Loop] Tool call: {tool_name} action={action}")
                
                try:
                    result = await computer_tool(
                        **cast(dict[str, Any], tool_use_block.get("input", {}))
                    )
                except ToolError as e:
                    result = ToolResult(error=e.message)
                    print(f"[Loop] Tool error: {e.message}")
                except Exception as e:
                    result = ToolResult(error=str(e))
                    print(f"[Loop] Tool exception: {e}")
                
                tool_result_content.append(
                    _make_api_tool_result(result, tool_use_block["id"])
                )
                
                if tool_output_callback:
                    tool_output_callback(result, tool_use_block["id"])
        
        # If no tool calls, the model is done
        if not tool_result_content:
            print(f"[Loop] No tool calls - model finished")
            _print_cost_summary(total_input_tokens, total_output_tokens, model)
            return messages
        
        print(f"[Loop] Continuing with {len(tool_result_content)} tool results")
        
        messages.append({"content": tool_result_content, "role": "user"})
    
    # Max iterations reached
    print(f"[Loop] Max iterations ({max_iterations}) reached")
    _print_cost_summary(total_input_tokens, total_output_tokens, model)
    return messages


def _response_to_params(response: BetaMessage) -> list[BetaContentBlockParam]:
    """Convert API response to content block params."""
    res: list[BetaContentBlockParam] = []
    
    for block in response.content:
        if isinstance(block, BetaTextBlock):
            if block.text:
                res.append(BetaTextBlockParam(type="text", text=block.text))
        elif isinstance(block, BetaToolUseBlock):
            # Tool use blocks - only include required fields to avoid sending
            # internal fields like 'caller' that the API doesn't accept on input
            res.append(cast(BetaToolUseBlockParam, {
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            }))
    
    return res


def _make_api_tool_result(
    result: ToolResult,
    tool_use_id: str,
) -> BetaToolResultBlockParam:
    """Convert a ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False
    
    if result.error:
        is_error = True
        tool_result_content = result.error
    else:
        if result.output:
            tool_result_content.append({
                "type": "text",
                "text": result.output,
            })
        if result.base64_image:
            tool_result_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": result.base64_image,
                },
            })
    
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }
