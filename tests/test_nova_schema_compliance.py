"""Tests for NOVA schema strict-mode compliance and evaluation correctness.

Tests cover:
1. Schema strict mode compliance (additionalProperties, all props required)
2. evaluate() async/sync correctness
3. diagnosis.py import path (no broken imports)
4. Single-turn prompt includes "continue" field
5. validate_nova_response with "reasoning" field
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Import examples/nova/src as package "src"
REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_NOVA_ROOT = REPO_ROOT / "examples" / "nova"
if str(EXAMPLE_NOVA_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_NOVA_ROOT))

from src.schemas import NOVA_SCHEMA
from src.schemas import validate_nova_response

# =====================================================================
# 1. Schema strict mode compliance
# =====================================================================


class TestSchemaStrictCompliance:
    """Verify NOVA_SCHEMA is valid for OpenAI strict structured outputs."""

    def _get_inner_schema(self) -> dict:
        return NOVA_SCHEMA["json_schema"]["schema"]

    def test_strict_is_true(self) -> None:
        assert NOVA_SCHEMA["json_schema"]["strict"] is True

    def test_top_level_has_additional_properties_false(self) -> None:
        schema = self._get_inner_schema()
        assert schema["additionalProperties"] is False

    def test_all_top_level_properties_are_required(self) -> None:
        """In strict mode, every property must be in required."""
        schema = self._get_inner_schema()
        props = set(schema["properties"].keys())
        required = set(schema["required"])
        missing = props - required
        assert not missing, f"Properties not in required (strict mode violation): {missing}"

    def test_reasoning_removed_from_schema(self) -> None:
        """'reasoning' should not be in properties or required
        (removed to fix strict mode issues)."""
        schema = self._get_inner_schema()
        assert "reasoning" not in schema.get("properties", {})
        assert "reasoning" not in schema.get("required", [])

    def _check_nested_objects(self, schema: dict, path: str = "root") -> list[str]:
        """Recursively check all nested objects have additionalProperties: false."""
        errors = []
        if schema.get("type") == "object":
            if "additionalProperties" not in schema:
                errors.append(f"{path}: missing additionalProperties")
            elif schema["additionalProperties"] is not False:
                errors.append(f"{path}: additionalProperties is not false")

            # Check all properties are required
            props = set(schema.get("properties", {}).keys())
            required = set(schema.get("required", []))
            missing = props - required
            if missing:
                errors.append(f"{path}: properties not in required: {missing}")

            for prop_name, prop_schema in schema.get("properties", {}).items():
                errors.extend(self._check_nested_objects(prop_schema, f"{path}.{prop_name}"))

        if schema.get("type") == "array" and "items" in schema:
            errors.extend(self._check_nested_objects(schema["items"], f"{path}[]"))

        return errors

    def test_all_nested_objects_strict_compliant(self) -> None:
        """Every nested object must have additionalProperties: false and all props required."""
        schema = self._get_inner_schema()
        errors = self._check_nested_objects(schema)
        assert not errors, "Strict mode violations:\n" + "\n".join(errors)


# =====================================================================
# 2. validate_nova_response with reasoning field
# =====================================================================


class TestValidateNovaResponse:
    """Ensure validation doesn't require 'reasoning' (it's schema-only)."""

    def _make_valid_response(self) -> dict:
        return {
            "caption": {
                "description": "Axial T2 showing lesion",
                "sequence_characteristics": "T2",
                "orientation": "axial",
                "confidence": 0.9,
                "findings": ["lesion"],
                "anatomical_regions": ["temporal lobe"],
            },
            "diagnosis": {
                "primary_diagnosis": "glioma",
                "differential_diagnoses": [],
                "confidence": 0.8,
                "evidence": ["mass effect"],
                "clinical_recommendations": "biopsy",
            },
            "localization": {
                "localizations": [
                    {
                        "finding": "mass",
                        "bounding_box": [10, 10, 50, 50],
                        "anatomical_location": "temporal",
                        "confidence": 0.7,
                    }
                ],
                "image_dimensions": {"width": 256, "height": 256},
                "coordinate_system": "absolute_pixels",
            },
            "continue": False,
            "reasoning": "Chain of thought here",
        }

    def test_valid_response_passes(self) -> None:
        assert validate_nova_response(self._make_valid_response()) is True

    def test_missing_continue_fails(self) -> None:
        resp = self._make_valid_response()
        del resp["continue"]
        assert validate_nova_response(resp) is False

    def test_missing_caption_fails(self) -> None:
        resp = self._make_valid_response()
        del resp["caption"]
        assert validate_nova_response(resp) is False

    def test_bad_caption_type_fails(self) -> None:
        resp = self._make_valid_response()
        resp["caption"] = "not a dict"
        assert validate_nova_response(resp) is False


# =====================================================================
# 3. diagnosis.py import path
# =====================================================================


class TestDiagnosisImports:
    """Verify diagnosis.py doesn't use broken import paths."""

    def test_no_broken_model_import(self) -> None:
        """The old 'from ..models import get_model_client' should be gone."""
        diagnosis_path = EXAMPLE_NOVA_ROOT / "src" / "evaluation" / "diagnosis.py"
        content = diagnosis_path.read_text()
        assert "from ..models import" not in content, (
            "diagnosis.py still uses 'from ..models import' which references a nonexistent module"
        )

    def test_uses_openai_directly(self) -> None:
        """Should use openai.AsyncOpenAI directly."""
        diagnosis_path = EXAMPLE_NOVA_ROOT / "src" / "evaluation" / "diagnosis.py"
        content = diagnosis_path.read_text()
        assert "AsyncOpenAI" in content


# =====================================================================
# 4. Single-turn prompt includes "continue"
# =====================================================================


class TestSingleTurnPrompt:
    """Verify single-turn prompt template includes required schema fields."""

    def test_continue_field_in_template(self) -> None:
        template_path = EXAMPLE_NOVA_ROOT / "src" / "prompts" / "single_turn" / "task.jinja"
        content = template_path.read_text()
        assert '"continue"' in content, (
            "Single-turn task.jinja missing 'continue' field in JSON example"
        )

    def test_reasoning_field_removed_from_template(self) -> None:
        """'reasoning' was removed from schema — template should not include it."""
        template_path = EXAMPLE_NOVA_ROOT / "src" / "prompts" / "single_turn" / "task.jinja"
        content = template_path.read_text()
        assert '"reasoning"' not in content, (
            "Single-turn task.jinja still contains 'reasoning' field — should be removed"
        )


# =====================================================================
# 5. evaluate() async wrapper
# =====================================================================


class TestEvaluateAsyncWrapper:
    """Verify evaluate_async is properly async and evaluate() is sync wrapper."""

    def test_evaluate_async_is_coroutine_function(self) -> None:
        import asyncio

        try:
            from src.evaluation import evaluate_async
        except (ImportError, ModuleNotFoundError):
            pytest.skip("torch not installed")
            return

        assert asyncio.iscoroutinefunction(evaluate_async)

    def test_evaluate_is_sync(self) -> None:
        import asyncio

        try:
            from src.evaluation import evaluate
        except (ImportError, ModuleNotFoundError):
            pytest.skip("torch not installed")
            return

        assert not asyncio.iscoroutinefunction(evaluate)


# =====================================================================
# 6. Reward function reads schema-aligned fields
# =====================================================================


class TestRewardSchemaAlignment:
    """Verify reward function correctly reads the strict-mode schema fields."""

    def test_reward_handles_new_required_fields(self) -> None:
        """Completions with all required fields (including new ones) should work."""
        from src.rewards import NOVAVerifiersReward

        completion = json.dumps(
            {
                "caption": {
                    "description": "left temporal lobe lesion",
                    "sequence_characteristics": "T2",
                    "orientation": "axial",
                    "confidence": 0.9,
                    "findings": ["lesion"],
                    "anatomical_regions": ["temporal lobe"],
                },
                "diagnosis": {
                    "primary_diagnosis": "glioma",
                    "differential_diagnoses": [],
                    "confidence": 0.8,
                    "evidence": ["left temporal lesion"],
                    "clinical_recommendations": "biopsy recommended",
                },
                "localization": {
                    "localizations": [
                        {
                            "finding": "lesion",
                            "bounding_box": [10, 10, 20, 20],
                            "anatomical_location": "left temporal lobe",
                            "confidence": 0.8,
                        }
                    ],
                    "image_dimensions": {"width": 64, "height": 64},
                    "coordinate_system": "absolute_pixels",
                },
                "continue": False,
                "reasoning": "Based on imaging findings...",
            }
        )
        info = {
            "caption": "left temporal lobe lesion",
            "diagnosis": "glioma",
            "boxes": [[10, 10, 20, 20]],
        }

        reward = NOVAVerifiersReward(task="all")
        score = reward("", completion, info)
        assert score == 1.0, f"Expected 1.0 for perfect match, got {score}"
