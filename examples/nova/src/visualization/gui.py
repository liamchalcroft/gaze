"""Streamlit GUI for NOVA predictions and ablation study visualization."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nest_asyncio
import pandas as pd
import streamlit as st
from beartype import beartype
from datasets import load_dataset
from PIL import Image

from src.processor import NOVAAgenticProcessor
from src.utils.confidence_calibration_utils import load_calibration_data_from_files

from .plotting import overlay_boxes
from .plotting import plot_ablation_comparison

# Allow nested event loops for Streamlit compatibility
# Must be called after imports but before any async code runs
nest_asyncio.apply()


@st.cache_data(show_spinner=False)
def load_nova_dataset() -> Any:
    """Load NOVA test split via Hugging Face."""
    return load_dataset("c-i-ber/Nova", split="train", trust_remote_code=False)


@beartype
async def predict_with_model(model_name: str, image: Image.Image, task: str) -> dict[str, Any]:
    """Run NOVA agentic processor for a single image."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        image.save(temp_file.name, format="JPEG", quality=95)
        temp_path = Path(temp_file.name)

    processor = NOVAAgenticProcessor(
        model_name=model_name,
        use_tools=True,
        use_web_search=False,
        max_turns=8,
        reasoning_enabled=True,
    )

    try:
        result = await processor.analyze(
            image_path=temp_path,
            metadata={"requested_task": task},
        )
        return {
            "model_name": model_name,
            "task": task,
            "response": result.final_response or {},
            "confidence": result.confidence,
            "num_turns": result.num_turns,
            "tool_calls": result.tool_call_count,
            "tools_used": list(result.get_tools_used()),
            "total_tokens": result.total_tokens,
        }
    finally:
        temp_path.unlink(missing_ok=True)


@beartype
def render_prediction(result: dict[str, Any], image_path: Path, task: str) -> None:
    """Render prediction outputs based on task."""
    response = result.get("response", {})

    if task == "localization":
        localizations = response.get("localization", {}).get("localizations", [])
        boxes = []
        labels = []
        for loc in localizations:
            bbox = loc.get("bounding_box") or loc.get("bbox")
            if bbox:
                boxes.append(bbox)
                labels.append(str(loc.get("label", "")))

        img = overlay_boxes(image_path, boxes, labels)
        st.image(img, caption=f"{task.capitalize()} result")
    elif task == "caption":
        caption = response.get("caption", {})
        text = caption.get("description") or caption.get("text") or caption or ""
        st.markdown(f"**Caption:** {text}")
    elif task == "diagnosis":
        diagnosis = response.get("diagnosis", {})
        diag_text = (
            diagnosis.get("primary_diagnosis")
            or diagnosis.get("diagnosis")
            or diagnosis.get("text")
            or ""
        )
        st.markdown(f"**Diagnosis:** {diag_text}")
    else:
        st.warning(f"Unsupported task: {task}")
        return

    with st.expander("Reasoning"):
        st.write(response.get("reasoning", "No reasoning available."))

    tools_used = result.get("tools_used")
    if tools_used:
        st.write(f"Tools used: {', '.join(tools_used)}")
    st.write(f"Confidence: {result.get('confidence', 0):.2f}")


@st.cache_data(show_spinner=False)
def load_ablation_results(results_dir: str) -> dict[str, Any]:
    """Load ablation study results for visualization."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return {}

    # Load calibration data
    calibration_data = load_calibration_data_from_files(results_path, "*_results.json")

    # Load evaluation metrics if available
    eval_metrics_path = results_path / "evaluation_metrics.json"
    eval_metrics = {}
    if eval_metrics_path.exists():
        with open(eval_metrics_path) as f:
            eval_metrics = json.load(f)

    return {
        "calibration_data": calibration_data,
        "evaluation_metrics": eval_metrics,
        "results_dir": results_path,
    }


def render_ablation_dashboard() -> None:
    """Render ablation study dashboard."""
    st.header("📊 Ablation Study Dashboard")

    # Sidebar for configuration
    st.sidebar.header("Ablation Configuration")
    results_dir = st.sidebar.text_input(
        "Results Directory",
        value="./results/ablation_studies",
        help="Directory containing ablation study results",
    )

    if not results_dir:
        st.warning("Please enter a results directory path")
        return

    # Load results
    ablation_data = load_ablation_results(results_dir)
    if not ablation_data:
        st.error(f"No results found in {results_dir}")
        return

    calibration_data = ablation_data["calibration_data"]
    eval_metrics = ablation_data["evaluation_metrics"]

    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Performance", "🎯 Calibration", "🛠️ Tool Usage", "📋 Raw Data"]
    )

    with tab1:
        render_performance_dashboard(eval_metrics)

    with tab2:
        render_calibration_dashboard(calibration_data)

    with tab3:
        render_tool_usage_dashboard(calibration_data)

    with tab4:
        render_raw_data_dashboard(ablation_data)


@beartype
def render_performance_dashboard(eval_metrics: dict[str, Any]) -> None:
    """Render performance comparison dashboard."""
    st.subheader("Performance Comparison")

    if not eval_metrics:
        st.info("No evaluation metrics available")
        return

    # Create performance comparison table
    config_data = []
    for config_name, metrics in eval_metrics.items():
        if isinstance(metrics, dict):
            config_data.append(
                {
                    "Configuration": config_name.replace("_", " ").title(),
                    "Accuracy": metrics.get("accuracy", 0),
                    "Confidence": metrics.get("confidence", 0),
                    "Tokens": metrics.get("avg_tokens", 0),
                }
            )

    if config_data:
        df = pd.DataFrame(config_data)
        st.dataframe(df, use_container_width=True)

        # Performance comparison chart
        st.pyplot(
            plot_ablation_comparison(
                eval_metrics,
                Path("temp_performance.png"),
                title="Configuration Performance Comparison",
            )
        )


@beartype
def render_calibration_dashboard(calibration_data: dict[str, Any]) -> None:
    """Render confidence calibration dashboard."""
    st.subheader("Confidence Calibration Analysis")

    if not calibration_data:
        st.info("No calibration data available")
        return

    # Configuration selector
    config_names = list(calibration_data.keys())
    selected_config = st.selectbox("Select Configuration", config_names)

    if selected_config:
        config_data = calibration_data[selected_config]

        # Basic calibration metrics
        col1, col2, col3 = st.columns(3)
        confidences = config_data.get("confidences", [])
        reliable_flags = config_data.get("reliable_flags", [])

        with col1:
            st.metric("Samples", len(confidences))
        with col2:
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            st.metric("Avg Confidence", f"{avg_conf:.3f}")
        with col3:
            reliable_rate = sum(reliable_flags) / len(reliable_flags) if reliable_flags else 0.0
            st.metric("Reliable Rate", f"{reliable_rate:.3f}")

        # Confidence level distribution
        if config_data.get("confidence_levels"):
            level_counts = {}
            for level in config_data["confidence_levels"]:
                level_counts[level] = level_counts.get(level, 0) + 1

            if level_counts:
                st.write("### Confidence Level Distribution")
                st.bar_chart(level_counts)

        # Reliability flag analysis
        if config_data.get("reliable_flags"):
            reliable_count = sum(config_data["reliable_flags"])
            total_count = len(config_data["reliable_flags"])
            unreliable_count = total_count - reliable_count

            st.write("### Reliability Flag Distribution")
            fig, ax = plt.subplots()
            ax.pie(
                [reliable_count, unreliable_count],
                labels=["Reliable", "Unreliable"],
                autopct="%1.1f%%",
                colors=["lightgreen", "lightcoral"],
            )
            st.pyplot(fig)


@beartype
def render_tool_usage_dashboard(calibration_data: dict[str, Any]) -> None:
    """Render tool usage analysis dashboard."""
    st.subheader("Tool Usage Analysis")

    if not calibration_data:
        st.info("No calibration data available")
        return

    # Check if any calibration data has tool usage information
    has_tool_data = any(config_data.get("sample_ids") for config_data in calibration_data.values())

    if not has_tool_data:
        st.info("Tool analysis available when research metrics are in results")
        return

    st.info("Tool usage data found - analysis implementation pending")


@beartype
def render_raw_data_dashboard(ablation_data: dict[str, Any]) -> None:
    """Render raw data exploration dashboard."""
    st.subheader("Raw Data Exploration")

    # File browser
    results_dir = ablation_data.get("results_dir")
    if results_dir and results_dir.exists():
        st.write(f"**Results Directory:** {results_dir}")

        # List result files
        result_files = list(results_dir.glob("*_results.json"))
        if result_files:
            st.write(f"**Result Files Found:** {len(result_files)}")

            # File selector
            selected_file = st.selectbox("Select Result File", [f.name for f in result_files])

            if selected_file:
                file_path = results_dir / selected_file
                try:
                    with open(file_path) as f:
                        file_data = json.load(f)

                    st.write(f"**File:** {selected_file}")
                    st.json(file_data)
                except (OSError, json.JSONDecodeError) as e:
                    st.error(f"Error loading file: {e}")

        # Evaluation metrics
        eval_metrics_path = results_dir / "evaluation_metrics.json"
        if eval_metrics_path.exists():
            with open(eval_metrics_path) as f:
                eval_metrics = json.load(f)

            st.write("**Evaluation Metrics**")
            st.json(eval_metrics)


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="NOVA VLM Explorer", layout="wide", initial_sidebar_state="expanded"
    )

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
    .stTabs [data-baseweb="tab"] {
        background-color: #e8f0fe;
        border-radius: 4px 4px 0 0;
    }
    </style>
    """
    st.markdown(pastel_css, unsafe_allow_html=True)

    # Main navigation
    st.title("🔬 NOVA VLM Explorer")
    st.markdown(
        "Interactive visualization and analysis platform for NOVA Retrieval VLM experiments"
    )

    # Mode selection
    mode = st.sidebar.selectbox(
        "Select Mode",
        ["Real-time Prediction", "Ablation Study Dashboard"],
        index=0,
        help="Choose between real-time model testing or ablation study analysis",
    )

    if mode == "Real-time Prediction":
        render_prediction_interface()
    else:
        render_ablation_dashboard()


def render_prediction_interface() -> None:
    """Render the original prediction interface."""
    st.header("🤖 Real-time Model Predictions")

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

    st.sidebar.header("Model Configuration")

    # Model selection
    model_option = st.sidebar.selectbox(
        "Model Selection Mode",
        ["Custom Models", "Ablation Configurations"],
        help="Choose between custom models or predefined ablation configurations",
    )

    if model_option == "Custom Models":
        # Original model input
        model_names: list[str] = []
        for i in range(3):
            name = st.sidebar.text_input(
                f"Model {i + 1}", value="" if i > 0 else "x-ai/grok-4.1-fast:free"
            )
            if name:
                model_names.append(name)
    else:
        # Ablation configuration selection
        ablation_configs = [
            "baseline_single_shot",
            "agentic_baseline",
            "reasoning_enabled",
            "no_visual_tools",
            "only_visual_tools",
            "no_tools",
            "limited_visual_tools",
            "with_retrieval",
        ]

        selected_configs = st.sidebar.multiselect(
            "Ablation Configurations", ablation_configs, default=["agentic_baseline"]
        )
        model_names = selected_configs

    # Task selection
    tasks = st.sidebar.multiselect(
        "Tasks",
        ["localization", "caption", "diagnosis"],
        default=["diagnosis"],
        help="Select tasks to run on the selected image",
    )

    # Enhanced options
    show_calibration = st.sidebar.checkbox("Show Confidence Calibration", value=True)
    show_reasoning = st.sidebar.checkbox("Show Reasoning Steps", value=True)

    if st.sidebar.button("🚀 Predict", type="primary"):
        if not model_names:
            st.sidebar.warning("Please select at least one model or configuration")
            return

        if not tasks:
            st.sidebar.warning("Please select at least one task")
            return

        for model in model_names:
            st.subheader(f"📊 Results for {model.replace('_', ' ').title()}")

            for task in tasks:
                with st.container():
                    st.markdown(f"### {task.capitalize()}")

                    try:
                        res = asyncio.run(predict_with_model(model, image, task))
                        render_prediction(res, image_path, task)

                        # Enhanced calibration display
                        if show_calibration and "confidence" in res:
                            col1, col2 = st.columns(2)
                            with col1:
                                confidence = res.get("confidence", 0.0)
                                st.metric("Confidence", f"{confidence:.3f}")
                            with col2:
                                st.metric("Turns", res.get("num_turns", 0))
                                st.metric("Tool Calls", res.get("tool_calls", 0))
                            st.caption(
                                f"Tokens: {res.get('total_tokens', 0)} | "
                                f"Tools used: {', '.join(res.get('tools_used', [])) or 'none'}"
                            )

                        # Enhanced reasoning display
                        if show_reasoning:
                            with st.expander("🧠 Detailed Reasoning", expanded=False):
                                reasoning = res.get("response", {}).get(
                                    "reasoning", "No detailed reasoning available"
                                )
                                st.write(reasoning)

                    except (OSError, ValueError) as e:
                        st.error(f"Error processing {model} for {task}: {e}")
                    # Note: RuntimeError should propagate to expose actual bugs

                    st.divider()


if __name__ == "__main__":
    main()
