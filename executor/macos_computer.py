"""macOS computer control tool using pyautogui."""

import asyncio
import base64
import logging
import math
import time
from dataclasses import dataclass, field
from enum import StrEnum
from io import BytesIO
from typing import Any, Literal

import pyautogui
from PIL import Image

# Set up logger for action debugging
logger = logging.getLogger(__name__)

# Configure pyautogui for safety
pyautogui.FAILSAFE = True  # Move mouse to corner to abort
pyautogui.PAUSE = 0.1  # Small pause between actions

# API constraints for images
MAX_LONG_EDGE = 1568
MAX_PIXELS = 1_150_000  # ~1.15 megapixels
JPEG_QUALITY = 80  # Good balance of quality and size


class ScalingSource(StrEnum):
    """Source of coordinates for scaling."""
    COMPUTER = "computer"  # Coordinates from the actual screen
    API = "api"  # Coordinates from Claude API


@dataclass(frozen=True)
class ToolResult:
    """Result from a tool execution."""
    
    output: str | None = None
    error: str | None = None
    base64_image: str | None = None
    
    def replace(self, **kwargs) -> "ToolResult":
        """Return a new ToolResult with replaced values."""
        return ToolResult(
            output=kwargs.get("output", self.output),
            error=kwargs.get("error", self.error),
            base64_image=kwargs.get("base64_image", self.base64_image),
        )


class ToolError(Exception):
    """Error during tool execution."""
    
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


# Action types
Action = Literal[
    "screenshot",
    "click",
    "double_click",
    "right_click",
    "middle_click",
    "mouse_move",
    "drag",
    "type",
    "key",
    "scroll",
    "wait",
    "cursor_position",
]


@dataclass
class MacOSComputerTool:
    """Tool for controlling macOS using pyautogui."""
    
    name: str = "computer"
    screenshot_delay: float = 0.5  # Delay before taking screenshot
    typing_interval: float = 0.02  # Interval between keystrokes
    action_delay: float = 0.3  # Delay after actions for UI to settle
    _scaling_enabled: bool = True  # Enable coordinate and image scaling
    
    # Screen dimensions (detected on init)
    width: int = field(init=False)
    height: int = field(init=False)
    _scale_factor: float = field(init=False)
    
    def __post_init__(self):
        """Initialize screen dimensions and calculate scale factor."""
        size = pyautogui.size()
        self.width = size.width
        self.height = size.height
        self._scale_factor = self._get_scale_factor(self.width, self.height)
    
    def _get_scale_factor(self, width: int, height: int) -> float:
        """Calculate scale factor to meet API constraints."""
        long_edge = max(width, height)
        total_pixels = width * height
        
        long_edge_scale = MAX_LONG_EDGE / long_edge
        total_pixels_scale = math.sqrt(MAX_PIXELS / total_pixels)
        
        return min(1.0, long_edge_scale, total_pixels_scale)
    
    def _get_scaled_dimensions(self) -> tuple[int, int]:
        """Get the scaled dimensions for screenshots."""
        if not self._scaling_enabled:
            return self.width, self.height
        return (
            int(self.width * self._scale_factor),
            int(self.height * self._scale_factor),
        )
    
    def scale_coordinates(
        self,
        source: ScalingSource,
        x: int,
        y: int,
    ) -> tuple[int, int]:
        """Scale coordinates between API space and screen space.
        
        Args:
            source: Where the coordinates are coming from.
            x: X coordinate.
            y: Y coordinate.
            
        Returns:
            Scaled (x, y) coordinates.
        """
        if not self._scaling_enabled or self._scale_factor == 1.0:
            return x, y
        
        if source == ScalingSource.API:
            # Scale up: API coordinates -> screen coordinates
            return round(x / self._scale_factor), round(y / self._scale_factor)
        else:
            # Scale down: screen coordinates -> API coordinates
            return round(x * self._scale_factor), round(y * self._scale_factor)
    
    def _log_action(self, action: str, params: dict[str, Any], result: ToolResult) -> None:
        """Log action for debugging per Anthropic best practices."""
        status = "error" if result.error else "success"
        # Filter out None values for cleaner logs
        filtered_params = {k: v for k, v in params.items() if v is not None}
        logger.info(f"Action: {action}, Params: {filtered_params}, Status: {status}")
    
    async def __call__(
        self,
        *,
        action: Action,
        coordinate: tuple[int, int] | list[int] | None = None,
        text: str | None = None,
        key: str | None = None,
        direction: str | None = None,
        amount: int | None = None,
        duration: float | None = None,
        start_coordinate: tuple[int, int] | list[int] | None = None,
        **kwargs,
    ) -> ToolResult:
        """Execute a computer control action.
        
        Args:
            action: The action to perform.
            coordinate: Target coordinates for mouse actions.
            text: Text to type.
            key: Key or key combination to press.
            direction: Scroll direction (up, down, left, right).
            amount: Scroll amount.
            duration: Wait duration in seconds.
            start_coordinate: Starting point for drag operations.
            
        Returns:
            ToolResult with output and/or screenshot.
        """
        # Collect params for logging
        params = {
            "coordinate": coordinate,
            "text": text,
            "key": key,
            "direction": direction,
            "amount": amount,
            "duration": duration,
            "start_coordinate": start_coordinate,
        }
        
        result: ToolResult
        try:
            if action == "screenshot":
                result = await self._screenshot()
            
            elif action == "cursor_position":
                pos = pyautogui.position()
                result = ToolResult(output=f"X={pos.x},Y={pos.y}")
            
            elif action == "click":
                result = await self._click(coordinate)
            
            elif action == "double_click":
                result = await self._click(coordinate, clicks=2)
            
            elif action == "right_click":
                result = await self._click(coordinate, button="right")
            
            elif action == "middle_click":
                result = await self._click(coordinate, button="middle")
            
            elif action == "mouse_move":
                result = await self._mouse_move(coordinate)
            
            elif action == "drag":
                result = await self._drag(start_coordinate, coordinate)
            
            elif action == "type":
                result = await self._type(text)
            
            elif action == "key":
                result = await self._key(key or text)
            
            elif action == "scroll":
                result = await self._scroll(direction, amount, coordinate)
            
            elif action == "wait":
                result = await self._wait(duration or 1.0)
            
            else:
                raise ToolError(f"Unknown action: {action}")
            
            # Log successful action
            self._log_action(action, params, result)
            return result
                
        except ToolError as e:
            # Log failed action
            error_result = ToolResult(error=e.message)
            self._log_action(action, params, error_result)
            raise
        except Exception as e:
            # Log failed action
            error_result = ToolResult(error=str(e))
            self._log_action(action, params, error_result)
            raise ToolError(f"Action failed: {e}")
    
    async def _screenshot(self) -> ToolResult:
        """Take a screenshot of the screen, resize and compress for API."""
        await asyncio.sleep(self.screenshot_delay)
        
        screenshot = pyautogui.screenshot()
        
        # Resize if scaling is enabled
        if self._scaling_enabled and self._scale_factor < 1.0:
            scaled_width, scaled_height = self._get_scaled_dimensions()
            screenshot = screenshot.resize(
                (scaled_width, scaled_height),
                Image.LANCZOS,
            )
        
        # Convert to JPEG for smaller file size
        buffer = BytesIO()
        # Convert to RGB (JPEG doesn't support RGBA)
        if screenshot.mode == "RGBA":
            screenshot = screenshot.convert("RGB")
        screenshot.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        base64_image = base64.b64encode(buffer.getvalue()).decode()
        
        return ToolResult(base64_image=base64_image)
    
    async def _click(
        self,
        coordinate: tuple[int, int] | list[int] | None,
        clicks: int = 1,
        button: str = "left",
    ) -> ToolResult:
        """Click at a coordinate."""
        if coordinate:
            x, y = self._validate_coordinate(coordinate)
            pyautogui.click(x, y, clicks=clicks, button=button)
        else:
            pyautogui.click(clicks=clicks, button=button)
        
        # Wait for UI to settle after click
        await asyncio.sleep(self.action_delay)
        return await self._screenshot()
    
    async def _mouse_move(
        self,
        coordinate: tuple[int, int] | list[int] | None,
    ) -> ToolResult:
        """Move mouse to a coordinate."""
        if not coordinate:
            raise ToolError("coordinate is required for mouse_move")
        
        x, y = self._validate_coordinate(coordinate)
        pyautogui.moveTo(x, y)
        
        return await self._screenshot()
    
    async def _drag(
        self,
        start: tuple[int, int] | list[int] | None,
        end: tuple[int, int] | list[int] | None,
    ) -> ToolResult:
        """Drag from start to end coordinate."""
        if not start or not end:
            raise ToolError("start_coordinate and coordinate are required for drag")
        
        start_x, start_y = self._validate_coordinate(start)
        end_x, end_y = self._validate_coordinate(end)
        
        pyautogui.moveTo(start_x, start_y)
        pyautogui.drag(end_x - start_x, end_y - start_y, duration=0.5)
        
        # Wait for UI to settle after drag
        await asyncio.sleep(self.action_delay)
        return await self._screenshot()
    
    async def _type(self, text: str | None) -> ToolResult:
        """Type text."""
        if not text:
            raise ToolError("text is required for type action")
        
        # Use write for typing (handles special characters better on macOS)
        pyautogui.write(text, interval=self.typing_interval)
        
        # Wait for UI to settle after typing
        await asyncio.sleep(self.action_delay)
        return await self._screenshot()
    
    async def _key(self, key: str | None) -> ToolResult:
        """Press a key or key combination."""
        if not key:
            raise ToolError("key is required for key action")
        
        # Handle key combinations like "cmd+c"
        if "+" in key:
            keys = [k.strip() for k in key.split("+")]
            # Map common key names
            mapped_keys = [self._map_key_name(k) for k in keys]
            pyautogui.hotkey(*mapped_keys)
        else:
            mapped_key = self._map_key_name(key)
            pyautogui.press(mapped_key)
        
        # Wait for UI to settle after key press
        await asyncio.sleep(self.action_delay)
        return await self._screenshot()
    
    async def _scroll(
        self,
        direction: str | None,
        amount: int | None,
        coordinate: tuple[int, int] | list[int] | None = None,
    ) -> ToolResult:
        """Scroll the screen."""
        if not direction:
            raise ToolError("direction is required for scroll")
        
        # Validate scroll amount
        if amount is not None:
            if not isinstance(amount, int) or amount < 0:
                raise ToolError("scroll amount must be a non-negative integer")
        
        amount = amount or 3
        
        # Move to coordinate if specified
        if coordinate:
            x, y = self._validate_coordinate(coordinate)
            pyautogui.moveTo(x, y)
        
        # Scroll based on direction
        if direction == "up":
            pyautogui.scroll(amount)
        elif direction == "down":
            pyautogui.scroll(-amount)
        elif direction == "left":
            pyautogui.hscroll(-amount)
        elif direction == "right":
            pyautogui.hscroll(amount)
        else:
            raise ToolError(f"Invalid scroll direction: {direction}")
        
        # Wait for UI to settle after scroll
        await asyncio.sleep(self.action_delay)
        return await self._screenshot()
    
    async def _wait(self, duration: float) -> ToolResult:
        """Wait for a specified duration."""
        # Validate duration per Anthropic best practices
        if duration < 0:
            raise ToolError("duration must be non-negative")
        if duration > 100:
            raise ToolError("duration is too long (max 100 seconds)")
        
        await asyncio.sleep(duration)
        return await self._screenshot()
    
    def _validate_coordinate(
        self,
        coordinate: tuple[int, int] | list[int],
    ) -> tuple[int, int]:
        """Validate and scale coordinates from API space to screen space."""
        if isinstance(coordinate, (list, tuple)) and len(coordinate) == 2:
            x, y = int(coordinate[0]), int(coordinate[1])
            
            # Scale coordinates from API space to screen space
            screen_x, screen_y = self.scale_coordinates(ScalingSource.API, x, y)
            
            if 0 <= screen_x <= self.width and 0 <= screen_y <= self.height:
                return screen_x, screen_y
            raise ToolError(f"Coordinates ({x}, {y}) -> ({screen_x}, {screen_y}) out of bounds")
        raise ToolError(f"Invalid coordinate format: {coordinate}")
    
    def _map_key_name(self, key: str) -> str:
        """Map common key names to pyautogui key names."""
        key_map = {
            "cmd": "command",
            "ctrl": "ctrl",
            "alt": "option",
            "option": "option",
            "shift": "shift",
            "enter": "return",
            "return": "return",
            "esc": "escape",
            "escape": "escape",
            "tab": "tab",
            "space": "space",
            "backspace": "backspace",
            "delete": "delete",
            "up": "up",
            "down": "down",
            "left": "left",
            "right": "right",
            "home": "home",
            "end": "end",
            "pageup": "pageup",
            "pagedown": "pagedown",
        }
        return key_map.get(key.lower(), key)
    
    def to_params(self) -> dict:
        """Return tool parameters for Claude API."""
        scaled_width, scaled_height = self._get_scaled_dimensions()
        return {
            "name": self.name,
            "type": "computer_20250124",
            "display_width_px": scaled_width,
            "display_height_px": scaled_height,
            "display_number": 1,
        }
    
    @property
    def options(self) -> dict:
        """Return tool options."""
        scaled_width, scaled_height = self._get_scaled_dimensions()
        return {
            "display_width_px": scaled_width,
            "display_height_px": scaled_height,
            "display_number": 1,
        }

