"""NOVA-specific agentic processor for brain-MRI analysis.

Extends AgenticProcessorBase with NOVA task prompts, schemas, and tool configuration.
Uses Jinja templates from nova/prompts/ for prompt generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from beartype import beartype

from radiant_harness import AgenticProcessorBase
from radiant_harness import ImageInput
from radiant_harness import Turn
from radiant_harness import create_prompt

from .schemas import NOVA_SCHEMA
from .schemas import validate_nova_response

# Path to NOVA prompts directory
NOVA_PROMPTS_DIR = Path(__file__).parent / "prompts"

# Confidence calculation constants
CONFIDENCE_BASE = 0.5
CONFIDENCE_COMPREHENSIVE_BONUS = 0.1
CONFIDENCE_PER_EVIDENCE = 0.02
CONFIDENCE_PER_DIFFERENTIAL = 0.02
CONFIDENCE_PER_LOCALIZATION = 0.02
CONFIDENCE_PER_TOOL_TURN = 0.05
CONFIDENCE_MAX_BONUS = 0.1


class NOVAAgenticProcessor(AgenticProcessorBase):
    """Agentic processor for NOVA brain-MRI benchmark tasks.

    Uses Jinja templates from nova/prompts/ for system and task prompts.
    Provides brain-MRI specific prompts and the NOVA unified schema.

    Example:
        processor = NOVAAgenticProcessor(
            model_name="openai/gpt-4o",
            use_tools=True,
            use_web_search=True,
            max_turns=10,
        )
        result = await processor.analyze(image_path, {"history": "..."})
    """

    DOMAIN = "Brain MRI"
    MODALITY = "MRI"
    BODY_REGION = "Brain"

    # Inherits _create_tool_registry from base class - uses standard tool
    # creation logic with self.use_tools and self.use_web_search flags

    def get_system_prompt(
        self,
        images: list[ImageInput],
        metadata: dict[str, Any],
    ) -> str:
        """Build NOVA-specific system prompt using Jinja templates."""
        history = metadata.get("history") or metadata.get("clinical_history") or ""
        # Get dimensions from first image if available
        width = images[0].width if images else 0
        height = images[0].height if images else 0
        image_id = images[0].path.name if images else "no_image"
        img_path = str(images[0].path) if images else ""

        context = {
            **metadata,
            "width": width,
            "height": height,
            "image_id": image_id,
            "img_path": img_path,
            "images": images,  # Pass full image list for multi-image support
            "enable_visual_tools": self.use_tools,
            "enable_web_search": self.use_web_search,
            "clinical_history": history,
            "has_images": len(images) > 0,
            "num_images": len(images),
            "policy": {
                "max_turns": self.max_turns,
                "requires_continue": True,
            },
        }

        return create_prompt(
            prompts_dir=NOVA_PROMPTS_DIR,
            mode="agentic",
            context=context,
            template_name="task.jinja",
        )

    def get_user_message(
        self,
        images: list[ImageInput],
        metadata: dict[str, Any],
    ) -> str:
        """Build NOVA-specific user message."""
        history = metadata.get("history") or metadata.get("clinical_history") or ""
        # Adapt message based on input mode
        if images:
            if len(images) == 1:
                context_parts = ["Analyze this brain MRI image comprehensively."]
            else:
                context_parts = [f"Analyze these {len(images)} brain MRI images comprehensively."]
                # Include image labels if available
                for i, img in enumerate(images):
                    label = img.label or f"Image {i + 1}"
                    context_parts.append(f"\n- {label}: {img.path.name}")
        else:
            context_parts = ["Based on the clinical information provided, assess the case."]

        if history:
            context_parts.append(f"\n**Clinical History:** {history}")

        if modality := metadata.get("modality"):
            context_parts.append(f"\n**Modality:** {modality}")

        if images:
            context_parts.append(
                "\nProvide complete captioning, diagnosis, and localization analysis."
            )
        else:
            context_parts.append("\nProvide your clinical assessment and differential diagnoses.")

        return "".join(context_parts)

    def get_response_schema(self) -> dict[str, Any] | None:
        """Return NOVA unified schema for structured outputs."""
        return NOVA_SCHEMA

    @beartype
    def validate_response(self, response: dict[str, Any]) -> bool:
        """Validate response has required NOVA fields."""
        return validate_nova_response(response)

    def calculate_confidence(
        self,
        response: dict[str, Any],
        turns: list[Turn],
    ) -> float:
        """Calculate confidence with NOVA-specific factors."""
        confidence = CONFIDENCE_BASE

        # Bonus for comprehensive response
        if all(field in response for field in ["caption", "diagnosis", "localization"]):
            confidence += CONFIDENCE_COMPREHENSIVE_BONUS

        # Bonus for evidence in diagnosis
        diagnosis = response.get("diagnosis", {})
        if evidence := diagnosis.get("evidence"):
            confidence += min(len(evidence) * CONFIDENCE_PER_EVIDENCE, CONFIDENCE_MAX_BONUS)

        # Bonus for differential diagnoses
        if differentials := diagnosis.get("differential_diagnoses"):
            confidence += min(
                len(differentials) * CONFIDENCE_PER_DIFFERENTIAL, CONFIDENCE_MAX_BONUS
            )

        # Bonus for localizations
        localization = response.get("localization", {})
        if localizations := localization.get("localizations"):
            confidence += min(
                len(localizations) * CONFIDENCE_PER_LOCALIZATION, CONFIDENCE_MAX_BONUS
            )

        # Bonus for tool usage
        tool_turns = sum(1 for t in turns if t.tool_calls)
        confidence += min(tool_turns * CONFIDENCE_PER_TOOL_TURN, CONFIDENCE_MAX_BONUS)

        return min(1.0, confidence)
