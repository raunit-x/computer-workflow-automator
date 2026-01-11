"""Parameter detection and suggestion for workflows."""

import json
import re
from typing import Any, TYPE_CHECKING

from .schema import Parameter, ParameterType, Workflow
from prompts.analyzer_prompts import PARAMETER_DETECTION_PROMPT

if TYPE_CHECKING:
    from utils.llm import LLMClient
    from config import ModelConfig


class ParameterDetector:
    """Detects and suggests parameters for workflows using LLMClient."""
    
    def __init__(
        self,
        model_config: "ModelConfig",
        llm_client: "LLMClient",
    ):
        """Initialize the detector.
        
        Args:
            model_config: ModelConfig with stage-specific model settings.
            llm_client: Unified LLMClient for API calls.
        """
        self.model_config = model_config
        self.llm = llm_client
    
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
        
        # Use LLMClient with parameter detection model
        content = [{"type": "text", "text": user_message}]
        data = self.llm.generate(
            model=self.model_config.parameter_detection,
            system_prompt=PARAMETER_DETECTION_PROMPT,
            content=content,
            max_tokens=4096,
            parse_json=True,
            json_type="object",
            default_json={},
            phase="parameter_detection",
        )
        
        # Convert to Parameter objects
        parameters = []
        existing_names = {p.name for p in workflow.parameters}
        
        for p in data.get("parameters", []):
            # Skip if already exists
            if p["name"] in existing_names:
                continue
            
            # Handle unknown parameter types gracefully
            try:
                param_type = ParameterType(p.get("type", "string"))
            except ValueError:
                param_type = ParameterType.STRING
            
            parameters.append(Parameter(
                name=p["name"],
                param_type=param_type,
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
