from pathlib import Path

from nova_retrieval_vlm.prompts.prompt_loader import load_prompt


def test_retrieval_localization_prompt(mock_image, mock_passages):
    prompt = load_prompt(
        "retrieval_localization.jinja",
        Path(mock_image),
        passages=mock_passages,
        metadata={"image_id": 1, "width": 100, "height": 100},
    )
    for passage in mock_passages:
        assert passage in prompt
