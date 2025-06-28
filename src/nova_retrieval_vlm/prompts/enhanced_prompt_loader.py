"""
Enhanced Prompt Loading with Jinja2 System Prompts

This module provides an enhanced prompt loading system that uses Jinja2 templates
for system prompts, following best practices for prompt management in GenAI applications.
Based on the approach described in the Jinja2 prompting guide.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from jinja2 import Environment, FileSystemLoader, Template
from loguru import logger

from .prompt_loader import load_jinja_template


class EnhancedPromptLoader:
    """
    Enhanced prompt loader that combines system prompts (Jinja templates) 
    with task-specific prompts for comprehensive medical image analysis.
    """
    
    def __init__(self, prompts_dir: Optional[Path] = None):
        """
        Initialize the enhanced prompt loader.
        
        Args:
            prompts_dir: Directory containing prompt templates. Defaults to module directory.
        """
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent
        
        self.prompts_dir = prompts_dir
        self.env = Environment(loader=FileSystemLoader(str(prompts_dir)))
        
        # Available system prompt modes
        self.system_modes = {
            'baseline': 'system/baseline.jinja',
            'multiturn': 'system/multiturn.jinja', 
            'visual': 'system/visual.jinja',
            'retrieval': 'system/retrieval.jinja',
            'web_search': 'system/web_search.jinja',
            'comprehensive': 'system/comprehensive.jinja',
        }
    
    def get_system_prompt(
        self, 
        mode: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Load and render a system prompt template for the specified mode.
        
        Args:
            mode: The analysis mode ('baseline', 'multiturn', 'visual', etc.)
            context: Additional context variables for the template
            
        Returns:
            Rendered system prompt string
        """
        if mode not in self.system_modes:
            raise ValueError(f"Unknown mode: {mode}. Available modes: {list(self.system_modes.keys())}")
        
        template_name = self.system_modes[mode]
        context = context or {}
        
        try:
            template = self.env.get_template(template_name)
            return template.render(**context)
        except Exception as e:
            logger.error(f"Failed to load system prompt template {template_name}: {e}")
            # Fallback to basic system prompt
            return self._get_fallback_system_prompt(mode)
    
    def create_enhanced_prompt(
        self,
        template_name: str,
        image_path: Path,
        passages: List[str],
        metadata: Dict[str, Any],
        mode: Optional[str] = None,
        system_context: Optional[Dict[str, Any]] = None,
        system_prompt_override: Optional[str] = None,
    ) -> str:
        """
        Create an enhanced prompt combining system prompt with task-specific prompt.
        
        Args:
            template_name: Name of the task-specific Jinja template
            image_path: Path to the image file
            passages: List of retrieved passages
            metadata: Dictionary of metadata
            mode: Analysis mode (auto-detected if None)
            system_context: Additional context for system prompt
            system_prompt_override: Custom system prompt to override default
            
        Returns:
            Combined enhanced prompt string
        """
        # Auto-detect mode if not provided
        if mode is None:
            mode = self._detect_mode_from_template(template_name)
        
        # Get system prompt
        if system_prompt_override:
            system_prompt = system_prompt_override
        else:
            system_context = system_context or {}
            # Add retrieval context if passages are provided
            if passages:
                system_context['use_retrieval'] = True
            system_prompt = self.get_system_prompt(mode, system_context)
        
        # Load task-specific prompt
        task_prompt = load_jinja_template(template_name, image_path, passages, metadata)
        
        # Combine prompts based on mode
        return self._combine_prompts(system_prompt, task_prompt, mode)
    
    def _detect_mode_from_template(self, template_name: str) -> str:
        """
        Detect the appropriate mode from template name.
        
        Args:
            template_name: Name of the Jinja template
            
        Returns:
            Detected mode string
        """
        if template_name.startswith("multiturn/"):
            return "multiturn"
        elif template_name.startswith("visual_multiturn/"):
            return "visual"
        elif template_name.startswith("retrieval_"):
            return "retrieval"
        elif template_name.startswith("baseline/"):
            return "baseline"
        else:
            return "baseline"
    
    def _combine_prompts(self, system_prompt: str, task_prompt: str, mode: str) -> str:
        """
        Combine system prompt with task-specific prompt based on mode.
        
        Args:
            system_prompt: The system prompt
            task_prompt: The task-specific prompt
            mode: Analysis mode
            
        Returns:
            Combined prompt string
        """
        if mode == "baseline":
            return f"{system_prompt}\n\n{task_prompt}"
        
        elif mode == "multiturn":
            return f"{system_prompt}\n\n<task_instructions>\n{task_prompt}\n</task_instructions>\n\nBegin your multi-turn analysis process."
        
        elif mode == "visual":
            return f"{system_prompt}\n\n<visual_analysis_instructions>\n{task_prompt}\n</visual_analysis_instructions>\n\nBegin your visual analysis with appropriate operations."
        
        elif mode == "retrieval":
            return f"{system_prompt}\n\n<retrieval_analysis_instructions>\n{task_prompt}\n</retrieval_analysis_instructions>\n\nBegin your retrieval-augmented analysis."
        
        elif mode == "web_search":
            return f"{system_prompt}\n\n<web_search_instructions>\n{task_prompt}\n</web_search_instructions>\n\nBegin your web search-augmented analysis."
        
        elif mode == "comprehensive":
            return f"{system_prompt}\n\n<comprehensive_analysis_instructions>\n{task_prompt}\n</comprehensive_analysis_instructions>\n\nBegin your comprehensive analysis using all available capabilities."
        
        else:
            return f"{system_prompt}\n\n{task_prompt}"
    
    def _get_fallback_system_prompt(self, mode: str) -> str:
        """
        Get a fallback system prompt if template loading fails.
        
        Args:
            mode: Analysis mode
            
        Returns:
            Fallback system prompt string
        """
        base_prompt = """You are an expert medical image analysis AI assistant, specializing in brain MRI interpretation and diagnosis. You operate within the NOVA medical image analysis system.

You are analyzing brain MRI images to provide accurate medical assessments including captioning, diagnosis, and localization of abnormalities. Your analysis will be used by medical professionals, so accuracy and clinical relevance are paramount.

<core_capabilities>
You have the following core capabilities:
1. **Medical Image Analysis**: Analyze brain MRI images for anatomical structures, pathologies, and abnormalities
2. **Clinical Diagnosis**: Provide differential diagnoses based on imaging findings
3. **Anatomical Localization**: Identify and localize specific brain regions and structures
4. **Medical Captioning**: Generate detailed, clinically relevant descriptions of imaging findings
5. **Evidence-Based Reasoning**: Base all conclusions on visible imaging evidence and medical knowledge
</core_capabilities>

<analysis_guidelines>
When analyzing medical images, follow these critical guidelines:
1. **Clinical Accuracy**: All assessments must be medically accurate and evidence-based
2. **Comprehensive Analysis**: Examine all visible structures and regions systematically
3. **Differential Diagnosis**: Consider multiple possible diagnoses when appropriate
4. **Anatomical Precision**: Use correct anatomical terminology and precise localization
5. **Clinical Relevance**: Focus on findings that are clinically significant
6. **Uncertainty Acknowledgment**: Acknowledge limitations and uncertainties in your analysis
</analysis_guidelines>

<medical_ethics>
1. **Professional Responsibility**: Maintain the highest standards of medical analysis
2. **Patient Safety**: Prioritize patient safety in all assessments
3. **Clinical Context**: Consider the clinical context and implications of findings
4. **Limitations**: Clearly state limitations and recommend clinical correlation when needed
</medical_ethics>

<communication>
1. Use precise medical terminology appropriate for clinical practice
2. Be thorough but concise in your analysis
3. Structure your responses logically and systematically
4. Provide clear reasoning for your conclusions
5. Acknowledge any uncertainties or limitations in your assessment
</communication>

Remember: Your analysis may directly impact patient care decisions. Always prioritize accuracy, thoroughness, and clinical relevance in your assessments."""

        if mode == "multiturn":
            return base_prompt + "\n\nThis is a multi-turn analysis mode. Engage in iterative, step-by-step reasoning."
        elif mode == "visual":
            return base_prompt + "\n\nThis is a visual analysis mode. You can request visual operations and web searches."
        elif mode == "retrieval":
            return base_prompt + "\n\nThis is a retrieval-augmented analysis mode. Use retrieved medical information."
        elif mode == "web_search":
            return base_prompt + "\n\nThis is a web search-augmented analysis mode. Access current medical information."
        elif mode == "comprehensive":
            return base_prompt + "\n\nThis is a comprehensive analysis mode. Use all available capabilities."
        else:
            return base_prompt
    
    def list_available_modes(self) -> List[str]:
        """
        List all available system prompt modes.
        
        Returns:
            List of available mode names
        """
        return list(self.system_modes.keys())
    
    def validate_mode(self, mode: str) -> bool:
        """
        Validate if a mode is supported.
        
        Args:
            mode: Mode to validate
            
        Returns:
            True if mode is supported, False otherwise
        """
        return mode in self.system_modes


# Global instance for easy access
_enhanced_loader = None

def get_enhanced_loader() -> EnhancedPromptLoader:
    """
    Get the global enhanced prompt loader instance.
    
    Returns:
        EnhancedPromptLoader instance
    """
    global _enhanced_loader
    if _enhanced_loader is None:
        _enhanced_loader = EnhancedPromptLoader()
    return _enhanced_loader


# Convenience functions for backward compatibility
def load_enhanced_prompt(
    template_name: str,
    image_path: Path,
    passages: List[str],
    metadata: Dict[str, Any],
    mode: Optional[str] = None,
    system_context: Optional[Dict[str, Any]] = None,
    system_prompt_override: Optional[str] = None,
) -> str:
    """
    Load an enhanced prompt with system prompt integration.
    
    Args:
        template_name: Name of the task-specific Jinja template
        image_path: Path to the image file
        passages: List of retrieved passages
        metadata: Dictionary of metadata
        mode: Analysis mode (auto-detected if None)
        system_context: Additional context for system prompt
        system_prompt_override: Custom system prompt to override default
        
    Returns:
        Enhanced prompt string
    """
    loader = get_enhanced_loader()
    return loader.create_enhanced_prompt(
        template_name=template_name,
        image_path=image_path,
        passages=passages,
        metadata=metadata,
        mode=mode,
        system_context=system_context,
        system_prompt_override=system_prompt_override,
    )


def get_system_prompt(mode: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Get a system prompt for the specified mode.
    
    Args:
        mode: Analysis mode
        context: Additional context variables
        
    Returns:
        System prompt string
    """
    loader = get_enhanced_loader()
    return loader.get_system_prompt(mode, context) 