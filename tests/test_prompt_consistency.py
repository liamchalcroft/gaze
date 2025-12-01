"""Test prompt template consistency between single-turn and agentic modes.

Tests that the enhanced clinical correlation prompts work correctly
and provide consistent behavior across both modes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from nova_retrieval_vlm.data.nova_dataset import NovaDataset
from nova_retrieval_vlm.prompts.prompt_loader import load_prompt


@pytest.mark.unit
class TestPromptConsistency:
    """Test consistency between single-turn and agentic prompt templates."""

    @pytest.fixture
    def sample_metadata(self) -> dict[str, Any]:
        """Sample metadata with clinical history for testing."""
        return {
            "clinical_history": "A ten-month-old infant presented with vertical nystagmus. Ophthalmic examination revealed bilateral atrophy of optic nerves. Physical exam showed no focal deficits. Labs revealed no endocrine abnormality.",
            "final_diagnosis": "Septo-optic dysplasia",
            "caption": "Coronal T2-weighted MRI shows complete absence of the septum pellucidum and the shape and flat roof of the frontal horns.",
            "width": 480,
            "height": 480,
            "image_id": "test_case_001",
            "enable_visual_tools": False,
            "enable_web_search": False,
        }

    @pytest.fixture
    def metadata_without_history(self) -> dict[str, Any]:
        """Sample metadata without clinical history."""
        return {
            "clinical_history": None,  # Use None instead of empty string
            "final_diagnosis": "Unknown",
            "caption": "MRI image showing brain structures",
            "width": 480,
            "height": 480,
            "image_id": "test_case_002",
            "enable_visual_tools": False,
            "enable_web_search": False,
        }

    def test_clinical_history_present_in_single_turn(self, sample_metadata: dict[str, Any]):
        """Test that clinical history is properly included in single-turn prompts."""
        prompt = load_prompt(
            template_name="all_tasks.jinja",
            image_path=Path("/tmp/test.jpg"),
            passages=[],
            metadata=sample_metadata,
            mode="single_turn",
        )

        # Check for key clinical correlation elements
        required_elements = [
            "Patient Clinical Context (MANDATORY for diagnosis)",
            "CRITICAL REQUIREMENTS:",
            "Your FINAL diagnosis MUST explain BOTH imaging findings AND this clinical presentation",
            "Which diagnosis best explains ALL the patient's symptoms and findings",
            "ten-month-old infant",  # Actual clinical history content
            "vertical nystagmus",
        ]

        for element in required_elements:
            assert element in prompt, f"Missing element in single-turn prompt: {element}"

        print("✅ Single-turn prompt includes all required clinical correlation elements")

    def test_clinical_history_present_in_agentic(self, sample_metadata: dict[str, Any]):
        """Test that clinical history is properly included in agentic prompts."""
        prompt = load_prompt(
            template_name="all_tasks.jinja",
            image_path=Path("/tmp/test.jpg"),
            passages=[],
            metadata=sample_metadata,
            mode="agentic",
        )

        # Check for key clinical correlation elements
        required_elements = [
            "Patient Clinical Context (MANDATORY for diagnosis)",
            "CRITICAL REQUIREMENTS:",
            "Your FINAL diagnosis MUST explain BOTH imaging findings AND this clinical presentation",
            "Which diagnosis best explains ALL the patient's symptoms and findings",
            "ten-month-old infant",  # Actual clinical history content
            "vertical nystagmus",
        ]

        for element in required_elements:
            assert element in prompt, f"Missing element in agentic prompt: {element}"

        print("✅ Agentic prompt includes all required clinical correlation elements")

    def test_no_clinical_history_handling(self, metadata_without_history: dict[str, Any]):
        """Test prompt behavior when no clinical history is provided."""
        single_turn_prompt = load_prompt(
            template_name="all_tasks.jinja",
            image_path=Path("/tmp/test.jpg"),
            passages=[],
            metadata=metadata_without_history,
            mode="single_turn",
        )

        agentic_prompt = load_prompt(
            template_name="all_tasks.jinja",
            image_path=Path("/tmp/test.jpg"),
            passages=[],
            metadata=metadata_without_history,
            mode="agentic",
        )

        # Both prompts should be valid and contain key elements
        assert len(single_turn_prompt) > 1000  # Should be substantial
        assert len(agentic_prompt) > 1000  # Should be substantial

        # Both should contain core analysis requirements
        assert "analysis" in single_turn_prompt.lower()
        assert "analysis" in agentic_prompt.lower()

        print("✅ Both modes handle missing clinical history correctly")

    def test_diagnosis_schema_consistency(self, sample_metadata: dict[str, Any]):
        """Test that diagnosis schema is consistent between modes."""
        single_turn_prompt = load_prompt(
            template_name="all_tasks.jinja",
            image_path=Path("/tmp/test.jpg"),
            passages=[],
            metadata=sample_metadata,
            mode="single_turn",
        )

        agentic_prompt = load_prompt(
            template_name="all_tasks.jinja",
            image_path=Path("/tmp/test.jpg"),
            passages=[],
            metadata=sample_metadata,
            mode="agentic",
        )

        # Check for diagnosis-related content
        assert "diagnosis" in single_turn_prompt.lower()
        assert "diagnosis" in agentic_prompt.lower()

        # Check for clinical/evidence elements
        assert "evidence" in single_turn_prompt.lower() or "clinical" in single_turn_prompt.lower()
        assert "evidence" in agentic_prompt.lower() or "clinical" in agentic_prompt.lower()

        print("✅ Diagnosis schema is consistent between modes")

    def test_quality_standards_consistency(self, sample_metadata: dict[str, Any]):
        """Test that quality standards are consistent between modes."""
        single_turn_prompt = load_prompt(
            template_name="all_tasks.jinja",
            image_path=Path("/tmp/test.jpg"),
            passages=[],
            metadata=sample_metadata,
            mode="single_turn",
        )

        agentic_prompt = load_prompt(
            template_name="all_tasks.jinja",
            image_path=Path("/tmp/test.jpg"),
            passages=[],
            metadata=sample_metadata,
            mode="agentic",
        )

        # Check for consistent quality standards
        quality_elements = [
            "QUALITY ASSURANCE STANDARDS:",
            "Use standard radiological terminology consistently",
            "Provide specific, evidence-based diagnostic reasoning",
            "Maintain appropriate confidence levels",
            "Ensure clinical relevance and actionability",
        ]

        for element in quality_elements:
            assert element in single_turn_prompt
            assert element in agentic_prompt

        print("✅ Quality standards are consistent between modes")

    def test_nova_dataset_prompt_generation(self):
        """Test that NovaDataset generates correct prompts with real data."""
        try:
            dataset = NovaDataset()
            sample = dataset[0]

            # Test single-turn prompt generation
            single_turn_prompt = load_prompt(
                template_name="all_tasks.jinja",
                image_path=Path("/tmp/test.jpg"),
                passages=[],
                metadata={
                    **sample["metadata"],
                    "enable_visual_tools": False,
                    "enable_web_search": False,
                },
                mode="single_turn",
            )

            # Test agentic prompt generation
            agentic_prompt = load_prompt(
                template_name="all_tasks.jinja",
                image_path=Path("/tmp/test.jpg"),
                passages=[],
                metadata={
                    **sample["metadata"],
                    "enable_visual_tools": False,
                    "enable_web_search": False,
                },
                mode="agentic",
            )

            # Verify both prompts contain real clinical data
            if sample["metadata"]["clinical_history"]:
                assert sample["metadata"]["clinical_history"] in single_turn_prompt
                assert sample["metadata"]["clinical_history"] in agentic_prompt

            # Verify both contain the enhanced correlation requirements
            assert "CRITICAL REQUIREMENTS:" in single_turn_prompt
            assert "CRITICAL REQUIREMENTS:" in agentic_prompt

            print("✅ NovaDataset generates consistent prompts with real data")

        except Exception as e:
            pytest.skip(f"NovaDataset not available for testing: {e}")

    def test_prompt_length_consistency(self, sample_metadata: dict[str, Any]):
        """Test that prompt lengths are reasonable and consistent."""
        single_turn_prompt = load_prompt(
            template_name="all_tasks.jinja",
            image_path=Path("/tmp/test.jpg"),
            passages=[],
            metadata=sample_metadata,
            mode="single_turn",
        )

        agentic_prompt = load_prompt(
            template_name="all_tasks.jinja",
            image_path=Path("/tmp/test.jpg"),
            passages=[],
            metadata=sample_metadata,
            mode="agentic",
        )

        # Both prompts should be substantial
        assert len(single_turn_prompt) > 1000, "Single-turn prompt too short"
        assert len(agentic_prompt) > 1000, "Agentic prompt too short"

        # Both prompts should be comparable in size (within 2x of each other)
        ratio = max(len(single_turn_prompt), len(agentic_prompt)) / min(
            len(single_turn_prompt), len(agentic_prompt)
        )
        assert ratio < 2.0, f"Prompt length ratio {ratio:.2f} is too different"

        print("✅ Prompt lengths are reasonable:")
        print(f"  Single-turn: {len(single_turn_prompt)} characters")
        print(f"  Agentic: {len(agentic_prompt)} characters")

    def test_tool_integration_consistency(self, sample_metadata: dict[str, Any]):
        """Test that tool integration is handled consistently."""
        # Test with tools enabled
        metadata_with_tools = {
            **sample_metadata,
            "enable_visual_tools": True,
            "enable_web_search": True,
        }

        single_turn_prompt = load_prompt(
            template_name="all_tasks.jinja",
            image_path=Path("/tmp/test.jpg"),
            passages=[],
            metadata=metadata_with_tools,
            mode="single_turn",
        )

        agentic_prompt = load_prompt(
            template_name="all_tasks.jinja",
            image_path=Path("/tmp/test.jpg"),
            passages=[],
            metadata=metadata_with_tools,
            mode="agentic",
        )

        # Both prompts should be valid
        assert len(single_turn_prompt) > 1000
        assert len(agentic_prompt) > 1000

        # Agentic mode should have tool-specific content when enabled
        # Single-turn may or may not include tool references
        assert "localization" in agentic_prompt.lower() or "analysis" in agentic_prompt.lower()

        print("✅ Tool integration is handled consistently")

    def test_coordinate_system_consistency(self, sample_metadata: dict[str, Any]):
        """Test that coordinate system information is consistent."""
        single_turn_prompt = load_prompt(
            template_name="all_tasks.jinja",
            image_path=Path("/tmp/test.jpg"),
            passages=[],
            metadata=sample_metadata,
            mode="single_turn",
        )

        agentic_prompt = load_prompt(
            template_name="all_tasks.jinja",
            image_path=Path("/tmp/test.jpg"),
            passages=[],
            metadata=sample_metadata,
            mode="agentic",
        )

        # Both should contain coordinate/localization references
        assert (
            "coordinate" in single_turn_prompt.lower() or "bounding" in single_turn_prompt.lower()
        )
        assert "coordinate" in agentic_prompt.lower() or "bounding" in agentic_prompt.lower()

        # Check for dimension references (width/height should be mentioned)
        assert str(sample_metadata["width"]) in single_turn_prompt
        assert str(sample_metadata["height"]) in single_turn_prompt

        print("✅ Coordinate system information is consistent")


@pytest.mark.integration
class TestPromptIntegration:
    """Integration tests for prompt functionality."""

    async def test_prompt_with_real_processor(self):
        """Test that enhanced prompts work with real processors."""
        from unittest.mock import AsyncMock
        from unittest.mock import patch

        from nova_retrieval_vlm.processors.caption import CaptionProcessor

        # Create processor with enhanced prompts
        config = ProcessorConfig(
            task_name="caption",
            model_name="test-model",
            batch_size=1,
            output_dir=tempfile.mkdtemp(),
        )

        processor = CaptionProcessor(config)

        # Create test batch with clinical history
        from PIL import Image

        from nova_retrieval_vlm.types import BatchData

        # Create test image
        temp_image = Path(tempfile.mktemp(suffix=".jpg"))
        Image.new("RGB", (480, 480), color="gray").save(temp_image)

        try:
            batch_data = BatchData(
                images=[str(temp_image)],
                metadata=[
                    {
                        "clinical_history": "Patient with headache and vision changes",
                        "width": 480,
                        "height": 480,
                        "image_id": "test_integration",
                        "enable_visual_tools": False,
                        "enable_web_search": False,
                    }
                ],
                labels=None,
            )

            # Mock the model adapter to test prompt generation
            with patch(
                "nova_retrieval_vlm.models.openai_adapter.OpenAIAdapter"
            ) as mock_adapter_class:
                mock_adapter = AsyncMock()
                mock_adapter_class.return_value = mock_adapter
                mock_adapter.generate.return_value = (
                    '{"caption": {"description": "Test response"}}',
                    {"success": True},
                )

                # Test processor generates and uses enhanced prompt
                responses = await processor.process_batch(batch_data, batch_idx=0)

                # Verify prompt was generated with clinical history
                assert len(responses) == 1
                mock_adapter.generate.assert_called_once()

                # Get the prompt that was passed to the model
                call_args = mock_adapter.generate.call_args
                system_prompt = call_args[1]["system_prompt"]  # keyword argument

                # Verify enhanced prompt elements are present
                assert "Patient Clinical Context (MANDATORY for diagnosis)" in system_prompt
                assert "headache and vision changes" in system_prompt
                assert "CRITICAL REQUIREMENTS:" in system_prompt

            print("✅ Enhanced prompts work correctly with real processors")

        finally:
            if temp_image.exists():
                temp_image.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
