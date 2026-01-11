"""Unified LLM client with provider abstraction and cost tracking."""

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .tracking import CostTracker, detect_provider
from analyzer.json_utils import extract_json_from_response

if TYPE_CHECKING:
    from .logger import WorkflowLogger

# Retry configuration for rate limit errors
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 2.0  # seconds
MAX_RETRY_DELAY = 120.0  # seconds

# Rate limiting for Gemini free tier (5 requests per minute)
GEMINI_MIN_REQUEST_INTERVAL = 12.0  # seconds between requests (5 per minute = 1 per 12s)
GEMINI_BATCH_SIZE = 5  # Max parallel requests before waiting
GEMINI_BATCH_WAIT = 60.0  # Wait time after a batch of 5 requests (1 minute for rate reset)


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class LLMClient:
    """Unified LLM client with provider abstraction and built-in cost tracking.
    
    Supports Anthropic, OpenAI, and Gemini providers with automatic detection
    based on model name. Handles provider-specific content formatting internally.
    
    Example:
        >>> client = LLMClient(cost_tracker)
        >>> response = client.generate(
        ...     model="gemini-2.0-flash",
        ...     system_prompt="You are a helpful assistant.",
        ...     content=[{"type": "text", "text": "Hello!"}],
        ... )
        >>> print(response)  # Returns text string
        
        >>> data = client.generate(
        ...     model="claude-sonnet-4-5-20250929",
        ...     system_prompt="Extract JSON",
        ...     content=[...],
        ...     parse_json=True,
        ...     json_type="array",
        ... )
        >>> print(data)  # Returns parsed list
    """
    
    def __init__(
        self,
        cost_tracker: CostTracker,
        logger: "WorkflowLogger | None" = None,
    ):
        """Initialize the LLM client.
        
        Args:
            cost_tracker: CostTracker instance for tracking usage and costs.
            logger: Optional WorkflowLogger for logging API calls.
        """
        self.cost_tracker = cost_tracker
        self.logger = logger
        
        # Lazy-loaded provider clients
        self._anthropic_client: Any = None
        self._openai_client: Any = None
        self._gemini_client: Any = None
        
        # Rate limiting for Gemini
        self._last_gemini_request_time: float = 0.0
        self._gemini_request_count_in_window: int = 0
        self._gemini_window_start: float = 0.0
        
        # For parallel batch processing
        self._parallel_mode: bool = False
    
    def _get_anthropic_client(self) -> Any:
        """Get or create Anthropic client."""
        if self._anthropic_client is None:
            from anthropic import Anthropic
            self._anthropic_client = Anthropic()
        return self._anthropic_client
    
    def _get_openai_client(self) -> Any:
        """Get or create OpenAI client."""
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI()
        return self._openai_client
    
    def _get_gemini_client(self) -> Any:
        """Get or create Gemini client using the new google-genai package."""
        if self._gemini_client is None:
            import os
            from google import genai
            
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            self._gemini_client = genai.Client(api_key=api_key)
        return self._gemini_client
    
    def generate(
        self,
        model: str,
        system_prompt: str,
        content: list[dict],
        max_tokens: int = 4096,
        parse_json: bool = False,
        json_type: str = "object",
        default_json: Any = None,
        tqdm_bar: Any = None,
        phase: str | None = None,
    ) -> str | dict | list:
        """Generate a response from an LLM.
        
        Args:
            model: Model name (e.g., "claude-sonnet-4-5-20250929", "gemini-2.0-flash").
            system_prompt: System prompt for the model.
            content: List of content blocks in unified format:
                - Text: {"type": "text", "text": "..."}
                - Image: {"type": "image", "path": Path(...)} or
                         {"type": "image", "data": "base64...", "media_type": "image/png"}
            max_tokens: Maximum tokens for response.
            parse_json: If True, parse response as JSON.
            json_type: Type of JSON to parse ("object" or "array").
            default_json: Default value if JSON parsing fails.
            tqdm_bar: Optional tqdm progress bar to update with cumulative tokens.
            phase: Optional phase name for cost tracking.
            
        Returns:
            Response text, or parsed JSON if parse_json=True.
        """
        provider = detect_provider(model)
        
        if provider == "anthropic":
            response = self._call_anthropic(model, system_prompt, content, max_tokens)
        elif provider == "openai":
            response = self._call_openai(model, system_prompt, content, max_tokens)
        elif provider == "gemini":
            response = self._call_gemini(model, system_prompt, content, max_tokens)
        else:
            raise ValueError(f"Unknown provider for model: {model}")
        
        # Track cost
        self.cost_tracker.add_usage(
            response.input_tokens,
            response.output_tokens,
            model=model,
            phase=phase,
        )
        
        # Log with tqdm support
        if self.logger:
            self.logger.api(
                response.input_tokens,
                response.output_tokens,
                console=(tqdm_bar is None),
                tqdm_bar=tqdm_bar,
                cumulative_in=self.cost_tracker.total_input_tokens,
                cumulative_out=self.cost_tracker.total_output_tokens,
            )
        
        # Parse JSON if requested
        if parse_json:
            return extract_json_from_response(
                response.text,
                json_type=json_type,
                default=default_json,
            )
        
        return response.text
    
    def generate_batch_parallel(
        self,
        requests: list[dict],
        batch_size: int = GEMINI_BATCH_SIZE,
        batch_wait: float = GEMINI_BATCH_WAIT,
        progress_callback: Any = None,
    ) -> list[Any]:
        """Generate responses for multiple requests in parallel batches.
        
        Optimized for Gemini free tier rate limits by firing `batch_size`
        requests in parallel, then waiting before the next batch.
        
        Args:
            requests: List of request dicts with keys matching generate() params:
                      model, system_prompt, content, max_tokens, parse_json, 
                      json_type, default_json, phase
            batch_size: Number of requests to fire in parallel (default: 5 for Gemini)
            batch_wait: Seconds to wait between batches (default: 60 for rate reset)
            progress_callback: Optional callback(completed, total) for progress updates
            
        Returns:
            List of responses in the same order as requests.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        total = len(requests)
        results: list[Any] = [None] * total  # Pre-allocate to maintain order
        completed = 0
        
        # Enable parallel mode to skip per-request rate limiting
        self._parallel_mode = True
        
        try:
            # Process in batches
            for batch_start in range(0, total, batch_size):
                batch_end = min(batch_start + batch_size, total)
                batch_requests = requests[batch_start:batch_end]
                batch_indices = list(range(batch_start, batch_end))
                
                # Fire batch in parallel
                with ThreadPoolExecutor(max_workers=len(batch_requests)) as executor:
                    # Submit all requests in batch
                    future_to_idx = {}
                    for i, req in enumerate(batch_requests):
                        idx = batch_indices[i]
                        future = executor.submit(
                            self._generate_single,
                            req,
                        )
                        future_to_idx[future] = idx
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            results[idx] = future.result()
                        except Exception as e:
                            # Store error to be raised later
                            results[idx] = e
                        
                        completed += 1
                        if progress_callback:
                            progress_callback(completed, total)
                
                # Wait between batches (except for the last batch)
                if batch_end < total:
                    if self.logger:
                        remaining = total - batch_end
                        self.logger.info(
                            f"Batch complete ({batch_end}/{total}). "
                            f"Waiting {batch_wait:.0f}s for rate limit reset..."
                        )
                    time.sleep(batch_wait)
        finally:
            # Reset parallel mode
            self._parallel_mode = False
        
        # Check for errors and raise the first one
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                raise result
        
        return results
    
    def _generate_single(self, req: dict) -> Any:
        """Helper for generate_batch_parallel - generates a single response."""
        return self.generate(
            model=req["model"],
            system_prompt=req["system_prompt"],
            content=req["content"],
            max_tokens=req.get("max_tokens", 4096),
            parse_json=req.get("parse_json", False),
            json_type=req.get("json_type", "object"),
            default_json=req.get("default_json"),
            tqdm_bar=None,  # Don't update tqdm from parallel threads
            phase=req.get("phase"),
        )
    
    def _call_anthropic(
        self,
        model: str,
        system_prompt: str,
        content: list[dict],
        max_tokens: int,
    ) -> LLMResponse:
        """Call Anthropic API."""
        client = self._get_anthropic_client()
        
        # Convert content to Anthropic format
        anthropic_content = self._convert_content_for_anthropic(content)
        
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": anthropic_content}],
        )
        
        return LLMResponse(
            text=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=model,
        )
    
    def _call_openai(
        self,
        model: str,
        system_prompt: str,
        content: list[dict],
        max_tokens: int,
    ) -> LLMResponse:
        """Call OpenAI API."""
        client = self._get_openai_client()
        
        # Convert content to OpenAI format
        openai_content = self._convert_content_for_openai(content)
        
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": openai_content},
            ],
        )
        
        return LLMResponse(
            text=response.choices[0].message.content,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            model=model,
        )
    
    def _call_gemini(
        self,
        model: str,
        system_prompt: str,
        content: list[dict],
        max_tokens: int,
        skip_rate_limit: bool = False,
    ) -> LLMResponse:
        """Call Gemini API using the new google-genai package with retry logic."""
        from google.genai import types, errors as genai_errors
        
        client = self._get_gemini_client()
        
        # Rate limiting: ensure minimum interval between requests
        # Skip when called from parallel batch (rate limiting handled at batch level)
        if not skip_rate_limit and not self._parallel_mode:
            elapsed = time.time() - self._last_gemini_request_time
            if elapsed < GEMINI_MIN_REQUEST_INTERVAL:
                wait_time = GEMINI_MIN_REQUEST_INTERVAL - elapsed
                if self.logger:
                    self.logger.info(f"Rate limiting: waiting {wait_time:.1f}s before Gemini request")
                time.sleep(wait_time)
        
        # Convert content to Gemini format (list of Parts)
        gemini_content = self._convert_content_for_gemini(content)
        
        # Build config
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
        )
        
        # Retry logic for rate limit errors
        last_exception = None
        retry_delay = INITIAL_RETRY_DELAY
        
        for attempt in range(MAX_RETRIES):
            try:
                # Call the API
                response = client.models.generate_content(
                    model=model,
                    contents=gemini_content,
                    config=config,
                )
                
                # Update last request time for rate limiting
                self._last_gemini_request_time = time.time()
                
                # Extract token counts from usage metadata
                usage = response.usage_metadata
                input_tokens = usage.prompt_token_count if usage else 0
                output_tokens = usage.candidates_token_count if usage else 0
                
                return LLMResponse(
                    text=response.text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=model,
                )
                
            except genai_errors.ClientError as e:
                # Check if it's a rate limit error (429)
                error_str = str(e).lower()
                is_rate_limit = "429" in error_str or "quota" in error_str or "rate" in error_str
                
                if is_rate_limit:
                    # Rate limit error - retry with exponential backoff
                    last_exception = e
                    # Update last request time even on failure
                    self._last_gemini_request_time = time.time()
                    
                    if attempt < MAX_RETRIES - 1:
                        if self.logger:
                            self.logger.warning(
                                f"Rate limit hit, retrying in {retry_delay:.1f}s "
                                f"(attempt {attempt + 1}/{MAX_RETRIES})"
                            )
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
                    else:
                        raise
                else:
                    # For other client errors, don't retry
                    raise
            except Exception as e:
                # For other errors, don't retry
                raise
        
        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected error in Gemini API call")
    
    def _convert_content_for_anthropic(self, content: list[dict]) -> list[dict]:
        """Convert unified content format to Anthropic format."""
        result = []
        for block in content:
            if block["type"] == "text":
                result.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image":
                if "path" in block:
                    # Load image from path
                    image_path = Path(block["path"])
                    with open(image_path, "rb") as f:
                        image_data = base64.b64encode(f.read()).decode("utf-8")
                    media_type = self._get_media_type(image_path)
                else:
                    image_data = block["data"]
                    media_type = block.get("media_type", "image/png")
                
                result.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    }
                })
        return result
    
    def _convert_content_for_openai(self, content: list[dict]) -> list[dict]:
        """Convert unified content format to OpenAI format."""
        result = []
        for block in content:
            if block["type"] == "text":
                result.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image":
                if "path" in block:
                    image_path = Path(block["path"])
                    with open(image_path, "rb") as f:
                        image_data = base64.b64encode(f.read()).decode("utf-8")
                    media_type = self._get_media_type(image_path)
                else:
                    image_data = block["data"]
                    media_type = block.get("media_type", "image/png")
                
                result.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{image_data}"}
                })
        return result
    
    def _convert_content_for_gemini(self, content: list[dict]) -> list[Any]:
        """Convert unified content format to Gemini format using types.Part."""
        from google.genai import types
        
        result = []
        for block in content:
            if block["type"] == "text":
                result.append(types.Part.from_text(text=block["text"]))
            elif block["type"] == "image":
                if "path" in block:
                    image_path = Path(block["path"])
                    with open(image_path, "rb") as f:
                        image_data = f.read()
                    media_type = self._get_media_type(image_path)
                else:
                    image_data = base64.b64decode(block["data"])
                    media_type = block.get("media_type", "image/png")
                
                # Create Gemini image part using inline_data
                result.append(types.Part.from_bytes(
                    data=image_data,
                    mime_type=media_type,
                ))
        return result
    
    def _get_media_type(self, image_path: Path) -> str:
        """Get MIME type from image file extension."""
        suffix = image_path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return mime_types.get(suffix, "image/png")

