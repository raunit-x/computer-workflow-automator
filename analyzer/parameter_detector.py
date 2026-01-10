"""Parameter detection and suggestion for workflows."""

import json
import re
from typing import Any

from anthropic import Anthropic

from .schema import Parameter, ParameterType, Workflow


PARAMETER_DETECTION_PROMPT = """You are an expert at identifying parameterizable values in computer workflows.

Given a workflow with instructions, analyze which values should be parameterized to make the workflow reusable with different inputs.

Consider parameterizing:
1. Search queries and text inputs - these are often the main variables
2. Form field values - names, addresses, selections
3. File paths or names
4. Numeric values that might change
5. Selection choices (dropdown values, checkboxes)
6. URLs or web addresses that might vary

DO NOT parameterize:
1. UI navigation (which buttons to click)
2. Keyboard shortcuts (Cmd+C, etc.)
3. Fixed application behaviors
4. Static element locations

For each suggested parameter, provide:
- A clear name (snake_case)
- The type (string, number, boolean, selection)
- A description of what it's for
- The default value from the workflow
- Whether it's required

Output as JSON:
{
    "parameters": [
        {
            "name": "parameter_name",
            "type": "string|number|boolean|selection",
            "description": "What this parameter controls",
            "default": "value from workflow",
            "required": true,
            "options": ["option1", "option2"]  // only for selection type
        }
    ],
    "reasoning": "Brief explanation of parameter choices"
}"""


class ParameterDetector:
    """Detects and suggests parameters for workflows."""
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250929",
        use_openai: bool = False,
    ):
        """Initialize the detector.
        
        Args:
            api_key: API key. If not provided, uses ANTHROPIC_API_KEY or OPENAI_API_KEY env var.
            model: Model to use.
            use_openai: If True, use OpenAI API instead of Anthropic.
        """
        self.use_openai = use_openai
        self.model = model
        
        if use_openai:
            from openai import OpenAI
            self.client: Any = OpenAI(api_key=api_key) if api_key else OpenAI()
        else:
            self.client = Anthropic(api_key=api_key) if api_key else Anthropic()
    
    def detect_parameters(self, workflow: Workflow) -> list[Parameter]:
        """Detect suggested parameters for a workflow.
        
        Args:
            workflow: The workflow to analyze.
            
        Returns:
            List of suggested parameters.
        """
        # Build the analysis prompt with workflow content
        workflow_content = f"""
# Workflow: {workflow.name}

## Description
{workflow.description}

## Existing Parameters
{json.dumps([p.to_dict() for p in workflow.parameters], indent=2) if workflow.parameters else "None defined yet"}

## Instructions
{workflow.instructions}
"""
        
        user_message = f"Analyze this workflow and suggest parameters that are NOT already defined:\n\n{workflow_content}\n\nOutput ONLY the JSON object."
        
        if self.use_openai:
            response = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=4096,
                messages=[
                    {"role": "system", "content": PARAMETER_DETECTION_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            response_text = response.choices[0].message.content.strip()
        else:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=PARAMETER_DETECTION_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            response_text = response.content[0].text.strip()
        
        # Extract JSON
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            return []
        
        # Convert to Parameter objects
        parameters = []
        existing_names = {p.name for p in workflow.parameters}
        
        for p in data.get("parameters", []):
            # Skip if already exists
            if p["name"] in existing_names:
                continue
            
            parameters.append(Parameter(
                name=p["name"],
                param_type=ParameterType(p.get("type", "string")),
                description=p.get("description", ""),
                default_value=p.get("default"),
                required=p.get("required", True),
                options=p.get("options"),
            ))
        
        return parameters
    
    def find_placeholders(self, workflow: Workflow) -> list[str]:
        """Find {placeholder} patterns in workflow instructions."""
        pattern = r'\{([a-z_][a-z0-9_]*)\}'
        matches = re.findall(pattern, workflow.instructions, re.IGNORECASE)
        return list(set(matches))
    
    def suggest_from_placeholders(self, workflow: Workflow) -> list[Parameter]:
        """Create parameter suggestions from placeholders found in instructions."""
        placeholders = self.find_placeholders(workflow)
        existing_names = {p.name for p in workflow.parameters}
        
        suggestions = []
        for name in placeholders:
            if name not in existing_names:
                suggestions.append(Parameter(
                    name=name,
                    param_type=ParameterType.STRING,
                    description=f"Value for {{{name}}} placeholder",
                    default_value=None,
                    required=True,
                ))
        
        return suggestions
    
    def merge_parameters(
        self,
        workflow: Workflow,
        detected: list[Parameter],
        user_selections: dict[str, bool],
    ) -> Workflow:
        """Merge detected parameters into a workflow based on user selections.
        
        Args:
            workflow: Original workflow.
            detected: List of detected parameters.
            user_selections: Dict mapping parameter names to whether to include them.
            
        Returns:
            Updated workflow with selected parameters.
        """
        # Get selected parameters
        selected = [p for p in detected if user_selections.get(p.name, True)]
        
        # Merge into workflow parameters (avoid duplicates)
        existing_names = {p.name for p in workflow.parameters}
        for param in selected:
            if param.name not in existing_names:
                workflow.parameters.append(param)
                existing_names.add(param.name)
        
        return workflow
    
    def infer_parameter_type(self, value: Any) -> ParameterType:
        """Infer the parameter type from a value."""
        if isinstance(value, bool):
            return ParameterType.BOOLEAN
        elif isinstance(value, (int, float)):
            return ParameterType.NUMBER
        elif isinstance(value, str):
            return ParameterType.STRING
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            if all(isinstance(v, (int, float)) for v in value):
                return ParameterType.COORDINATE
        return ParameterType.STRING
    
    def create_parameter_from_value(
        self,
        name: str,
        value: Any,
        description: str = "",
    ) -> Parameter:
        """Create a parameter from a value with automatic type inference."""
        return Parameter(
            name=name,
            param_type=self.infer_parameter_type(value),
            description=description,
            default_value=value,
            required=True,
        )
