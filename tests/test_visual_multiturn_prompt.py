from pathlib import Path

from nova_retrieval_vlm.prompts.prompt_loader import load_prompt


def test_visual_multiturn_ops_prompt(mock_image):
    prompt = load_prompt(
        "visual_multiturn/ops_request.jinja",
        Path(mock_image),
    )
    assert "zoom_factor" in prompt
    assert "need_more_ops" in prompt
