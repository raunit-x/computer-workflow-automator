"""Agentic sampling loop for workflow execution."""

from collections.abc import Callable
from datetime import datetime
from typing import Any, cast, TYPE_CHECKING

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
from .reference_frames_tool import ReferenceFramesTool
from prompts.executor_prompts import WORKFLOW_SYSTEM_PROMPT

if TYPE_CHECKING:
    from utils.logger import WorkflowLogger
    from utils.tracking import CostTracker

# Retry threshold for auto-injecting reference frames
RETRY_THRESHOLD = 3


def _format_parameters_section(parameters: dict[str, Any]) -> str:
    """Format parameters for display in the system prompt."""
    if not parameters:
        return "No parameters provided - use default values from the workflow."
    
    lines = ["The following parameter values should be used:\n"]
    for name, value in parameters.items():
        lines.append(f"- **{name}**: `{value}`")
    
    lines.append("\nReplace any `{" + "parameter_name}` placeholders with these values.")
    return "\n".join(lines)


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
    iteration_callback: Callable[[int, int], None] | None = None,
    api_usage_callback: Callable[[int, int], None] | None = None,
    max_tokens: int = 16384,
    max_iterations: int = 100,
    logger: "WorkflowLogger | None" = None,
    cost_tracker: "CostTracker | None" = None,
    processed_session_path: str | None = None,
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
        iteration_callback: Callback for iteration progress (current, max).
        api_usage_callback: Callback for API token usage (input, output).
        max_tokens: Maximum tokens for responses.
        max_iterations: Maximum number of loop iterations.
        logger: Optional WorkflowLogger for structured output.
        cost_tracker: Optional CostTracker for cost accumulation.
        processed_session_path: Path to processed session for reference frames.
        
    Returns:
        List of conversation messages.
    """
    client = Anthropic(api_key=api_key) if api_key else Anthropic()
    computer_tool = MacOSComputerTool()
    
    # Initialize reference frames tool if session path is available
    reference_frames_tool = ReferenceFramesTool(processed_session_path=processed_session_path)
    reference_frames_tool.set_workflow_context(processed_session_path, workflow_instructions)
    reference_frames_available = processed_session_path is not None
    
    # Tracking for retry detection and reference frame injection
    consecutive_errors = 0
    last_action_description = ""
    reference_frames_injected = False
    
    # Track token usage for cost calculation
    total_input_tokens = 0
    total_output_tokens = 0
    
    # Helper for logging
    def log_info(msg: str) -> None:
        if logger:
            logger.info(msg)
        else:
            print(f"[Loop] {msg}")
    
    def log_step(msg: str) -> None:
        if logger:
            logger.step(msg)
        else:
            print(f"[Loop] {msg}")
    
    def log_error(msg: str) -> None:
        if logger:
            logger.error(msg)
        else:
            print(f"[Loop] ERROR: {msg}")
    
    def log_api(input_tok: int, output_tok: int) -> None:
        if logger:
            logger.api(input_tok, output_tok)
        else:
            print(f"[Loop] Tokens: {input_tok:,} in, {output_tok:,} out")
    
    def log_iteration(current: int, total: int) -> None:
        if logger:
            logger.iteration(current, total)
        else:
            print(f"[Loop] Iteration {current}/{total}")
    
    def log_tool(tool_name: str, action: str) -> None:
        if logger:
            logger.tool(tool_name, action)
        else:
            print(f"[Loop] Tool call: {tool_name} action={action}")
    
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
        log_iteration(iteration, max_iterations)
        
        if iteration_callback:
            iteration_callback(iteration, max_iterations)
        
        # Build tools list - always include computer, optionally include reference_frames
        tools_list: list[BetaToolUnionParam] = [
            cast(BetaToolUnionParam, computer_tool.to_params())
        ]
        if reference_frames_available:
            tools_list.append(cast(BetaToolUnionParam, reference_frames_tool.to_params()))
        
        try:
            log_step(f"Calling API with model: {model}")
            raw_response = client.beta.messages.with_raw_response.create(
                max_tokens=max_tokens,
                messages=messages,
                model=model,
                system=[system],
                tools=tools_list,
                betas=["computer-use-2025-01-24"],
            )
            log_info("API call succeeded")
        except (APIStatusError, APIResponseValidationError) as e:
            log_error(f"API Error: {e}")
            if api_response_callback:
                api_response_callback(e.request, e.response, e)
            raise  # Re-raise so WorkflowRunner knows it failed
        except APIError as e:
            log_error(f"API Error: {e}")
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
        
        log_info(f"Response has {len(response_params)} content blocks, stop_reason={response.stop_reason}")
        log_api(input_tokens, output_tokens)
        
        # Track costs
        if cost_tracker:
            cost_tracker.add_usage(input_tokens, output_tokens, model=model)
        
        if api_usage_callback:
            api_usage_callback(input_tokens, output_tokens)
        
        messages.append({
            "role": "assistant",
            "content": response_params,
        })
        
        # Process tool calls
        tool_result_content: list[BetaToolResultBlockParam] = []
        had_error_this_iteration = False
        
        for content_block in response_params:
            if output_callback:
                output_callback(content_block)
            
            if isinstance(content_block, dict) and content_block.get("type") == "tool_use":
                tool_use_block = cast(BetaToolUseBlockParam, content_block)
                tool_name = tool_use_block.get("name", "unknown")
                tool_input = tool_use_block.get("input", {})
                action = tool_input.get("action", "unknown") if isinstance(tool_input, dict) else "unknown"
                log_tool(tool_name, action)
                
                try:
                    if tool_name == "reference_frames":
                        # Handle reference_frames tool
                        result = await reference_frames_tool(
                            **cast(dict[str, Any], tool_use_block.get("input", {}))
                        )
                        log_info("Reference frames retrieved")
                    else:
                        # Handle computer tool
                        result = await computer_tool(
                            **cast(dict[str, Any], tool_use_block.get("input", {}))
                        )
                except ToolError as e:
                    result = ToolResult(error=e.message)
                    log_error(f"Tool error: {e.message}")
                    had_error_this_iteration = True
                except Exception as e:
                    result = ToolResult(error=str(e))
                    log_error(f"Tool exception: {e}")
                    had_error_this_iteration = True
                
                # Track current action for error detection
                if tool_name == "computer" and action != "screenshot":
                    current_action = f"{action}:{tool_input}"
                    if result.error:
                        if current_action == last_action_description:
                            consecutive_errors += 1
                        else:
                            consecutive_errors = 1
                        last_action_description = current_action
                    else:
                        consecutive_errors = 0
                        last_action_description = ""
                
                tool_result_content.append(
                    _make_api_tool_result(result, tool_use_block["id"])
                )
                
                if tool_output_callback:
                    tool_output_callback(result, tool_use_block["id"])
        
        # Auto-inject reference frames hint if stuck
        if (reference_frames_available and 
            not reference_frames_injected and
            consecutive_errors >= RETRY_THRESHOLD):
            log_info(f"Detected {consecutive_errors} consecutive errors - suggesting reference frames")
            reference_frames_injected = True
            # Add a hint message to suggest using reference frames
            hint_message: BetaToolResultBlockParam = {
                "type": "tool_result",
                "tool_use_id": "system_hint",
                "content": (
                    "⚠️ You appear to be stuck on this step after multiple attempts. "
                    "Consider using the `reference_frames` tool to see what the screen "
                    "looked like when the human performed this step in the original recording. "
                    "This may help you identify the correct UI elements or understand the expected state."
                ),
                "is_error": False,
            }
            # We can't add this directly as a tool result, but we can note it in logs
            # The prompt already mentions this capability
            log_info("Hint: Use reference_frames tool to see original recording")
        
        # If no tool calls, the model is done
        if not tool_result_content:
            log_info("No tool calls - model finished")
            return messages
        
        log_step(f"Continuing with {len(tool_result_content)} tool results")
        
        messages.append({"content": tool_result_content, "role": "user"})
    
    # Max iterations reached
    log_info(f"Max iterations ({max_iterations}) reached")
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
