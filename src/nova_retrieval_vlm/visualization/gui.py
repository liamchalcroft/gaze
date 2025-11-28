"""Streamlit GUI for comparing NOVA predictions across models."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from datasets import load_dataset
from PIL import Image

from nova_retrieval_vlm.models.openai_adapter import OpenAIAdapter
from nova_retrieval_vlm.models.openrouter_adapter import OpenRouterAdapter

from .plotting import overlay_boxes


@st.cache_data(show_spinner=False)
def load_nova_dataset() -> Any:
    """Load NOVA test split via Hugging Face."""
    return load_dataset("Ano-2090/Nova", split="test", trust_remote_code=False)


async def predict_with_model(model_name: str, image: Image.Image, task: str) -> dict[str, Any]:
    """Real prediction function using proper model adapters."""
    # Save image to temporary file for model adapters
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        image.save(temp_file.name, format="JPEG", quality=95)
        image_path = Path(temp_file.name)

    try:
        # Create model adapter based on model name
        if model_name.startswith("openai/"):
            adapter = OpenAIAdapter(model_name=model_name.split("/", 1)[1])
        else:
            adapter = OpenRouterAdapter(model_name=model_name)

        # Create task-specific prompt based on task type
        if task == "localization":
            system_prompt = (
                "Analyze this medical image and locate any abnormalities or regions of interest. "
                "Return your response as JSON with 'boxes' (list of [x, y, width, height] coordinates) "
                "and 'reasoning'."
            )
        elif task == "caption":
            system_prompt = (
                "Generate a detailed medical caption for this image. "
                "Return your response as JSON with 'text' (the caption) and 'reasoning'."
            )
        elif task == "diagnosis":
            system_prompt = (
                "Provide a medical diagnosis based on this image. "
                "Return your response as JSON with 'text' (diagnosis), 'confidence' (0-1), "
                "and 'reasoning'."
            )
        else:
            raise ValueError(f"Unknown task: {task}")

        # Get model response
        response_text, generation_log = await adapter.generate(
            image_path=image_path,
            passages=[],  # No retrieval for now - can be added later
            system_prompt=system_prompt,
            max_tokens=512,
            temperature=0.1,
        )

        # Parse JSON response
        try:
            result = json.loads(response_text)
            result["model_name"] = model_name
            result["generation_log"] = {
                "tokens": generation_log.tokens,
                "cost": generation_log.cost,
                "timestamp": generation_log.timestamp,
            }
            return result
        except json.JSONDecodeError:
            # If not JSON, return as text response
            return {
                "text": response_text,
                "reasoning": f"Response from {model_name}",
                "model_name": model_name,
                "generation_log": {
                    "tokens": generation_log.tokens,
                    "cost": generation_log.cost,
                    "timestamp": generation_log.timestamp,
                },
            }

    finally:
        # Clean up temporary file
        image_path.unlink(missing_ok=True)


def render_prediction(result: dict[str, Any], image_path: Path, task: str) -> None:
    """Render prediction outputs based on task."""
    if task == "localization":
        img = overlay_boxes(image_path, result.get("boxes", []), result.get("labels"))
        st.image(img, caption=f"{task.capitalize()} result")
    else:
        st.markdown(f"**{task.capitalize()}:** {result.get('text', '')}")
    with st.expander("Reasoning trace"):
        steps = result.get("reasoning_steps")
        if steps:
            for i, step in enumerate(steps, 1):
                with st.expander(f"Step {i}"):
                    st.write(step)
        else:
            st.write(result.get("reasoning", "No reasoning available."))
    with st.expander("Retrieved passages"):
        for passage in result.get("retrieval", []):
            st.markdown(f"- {passage}")


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="NOVA Demo", layout="wide")

    pastel_css = """
    <style>
    body {
        background-color: #f6f5f3;
        color: #333333;
    }
    .stApp {
        background-color: #f6f5f3;
    }
    .stButton>button {
        background-color: #c7dfe8;
        color: black;
        border-radius: 4px;
        padding: 0.25em 0.75em;
    }
    </style>
    """
    st.markdown(pastel_css, unsafe_allow_html=True)

    st.title("NOVA Prediction Explorer")
    ds = load_nova_dataset()
    sample_idx = st.number_input("Sample index", min_value=0, max_value=len(ds) - 1, value=0)
    record = ds[int(sample_idx)]
    image = (
        record["image"]
        if isinstance(record["image"], Image.Image)
        else Image.fromarray(record["image"])
    )
    image_path = Path(f"sample_{sample_idx}.png")
    image.save(image_path)
    st.image(image, caption=record.get("filename", f"Sample {sample_idx}"))

    st.sidebar.header("Models")
    model_names: list[str] = []
    for i in range(3):
        name = st.sidebar.text_input(f"Model {i + 1}", value="model-name" if i == 0 else "")
        if name:
            model_names.append(name)
    tasks = st.sidebar.multiselect(
        "Tasks", ["localization", "caption", "diagnosis"], default=["caption"]
    )

    if st.sidebar.button("Predict"):
        for model in model_names:
            st.subheader(f"Results for {model}")
            for task in tasks:
                with st.container():
                    st.markdown(f"### {task.capitalize()}")
                    res = asyncio.run(predict_with_model(model, image, task))
                    render_prediction(res, image_path, task)


if __name__ == "__main__":
    main()
