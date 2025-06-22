"""Streamlit GUI for comparing NOVA predictions across models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from datasets import load_dataset
from PIL import Image

from .plotting import overlay_boxes


@st.cache_data(show_spinner=False)
def load_nova_dataset() -> Any:
    """Load NOVA test split via Hugging Face."""
    return load_dataset("Ano-2090/Nova", split="test", trust_remote_code=False)


def dummy_predict(model_name: str, image: Image.Image, task: str) -> Dict[str, Any]:
    """Placeholder prediction function.

    This stub returns deterministic fake outputs so that the GUI remains
    functional in environments without model access.
    """
    if task == "localization":
        return {
            "boxes": [[20, 20, 80, 80]],
            "labels": ["anomaly"],
            "reasoning": f"{model_name} reasoning for localization.",
            "retrieval": ["Example guideline passage 1", "Example guideline passage 2"],
        }
    if task == "caption":
        return {
            "text": f"{model_name} caption describing the image.",
            "reasoning": f"{model_name} reasoning for captioning.",
            "retrieval": ["Example caption passage"],
        }
    if task == "diagnosis":
        return {
            "text": f"{model_name} predicted diagnosis.",
            "reasoning": f"{model_name} reasoning for diagnosis.",
            "retrieval": ["Example diagnosis passage"],
        }
    return {}


def render_prediction(result: Dict[str, Any], image_path: Path, task: str) -> None:
    """Render prediction outputs based on task."""
    if task == "localization":
        img = overlay_boxes(image_path, result.get("boxes", []), result.get("labels"))
        st.image(img, caption=f"{task.capitalize()} result")
    else:
        st.markdown(f"**{task.capitalize()}:** {result.get('text', '')}")
    with st.expander("Reasoning trace"):
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
    model_names: List[str] = []
    for i in range(3):
        name = st.sidebar.text_input(f"Model {i+1}", value="model-name" if i == 0 else "")
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
                    res = dummy_predict(model, image, task)
                    render_prediction(res, image_path, task)


if __name__ == "__main__":
    main()
