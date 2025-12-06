"""NOVA Brain-MRI Benchmark Implementation.

Reference implementation showing how to use the harness for a specific dataset.
This module demonstrates the minimal code needed to adapt the harness to a new
medical imaging benchmark.

The NOVA benchmark evaluates VLMs on brain MRI analysis across three tasks:
    - Captioning: Generating radiological descriptions
    - Diagnosis: Identifying primary diagnosis and differentials
    - Localization: Detecting and localizing abnormalities with bounding boxes

Implementation Structure:
    ```
    nova/
    ├── __init__.py       # This file - exports and documentation
    ├── processor.py      # NOVAAgenticProcessor
    ├── schemas.py        # NOVA_SCHEMA for structured output
    └── prompts/          # Jinja2 templates for NOVA tasks
        ├── agentic/
        │   ├── system.jinja
        │   └── all_tasks.jinja
        └── single_turn/
            └── ...
    ```

Creating Your Own Implementation:
    To create a processor for a new dataset, follow this pattern:

    1. **Create your module structure:**
       ```
       my_dataset/
       ├── __init__.py
       ├── processor.py
       ├── schemas.py
       └── prompts/
           └── agentic/
               ├── system.jinja
               └── task.jinja
       ```

    2. **Define your schema** (schemas.py):
       ```python
       MY_SCHEMA = {
           "type": "object",
           "properties": {
               "findings": {"type": "array", "items": {"type": "string"}},
               "diagnosis": {"type": "string"},
               "confidence": {"type": "number"},
               "continue": {"type": "boolean"},
           },
           "required": ["findings", "diagnosis", "continue"],
       }

       def validate_response(response: dict) -> bool:
           return all(k in response for k in ["findings", "diagnosis"])
       ```

    3. **Implement your processor** (processor.py):
       ```python
       from pathlib import Path
       from harness import AgenticProcessorBase, create_prompt

       PROMPTS_DIR = Path(__file__).parent / "prompts"

       class MyDatasetProcessor(AgenticProcessorBase):
           def get_system_prompt(self, image_path, metadata, width, height):
               return create_prompt(
                   prompts_dir=PROMPTS_DIR,
                   template_name="task.jinja",
                   mode="agentic",
                   context={"width": width, "height": height, **metadata},
               )

           def get_user_message(self, image_path, metadata):
               return f"Analyze this image. History: {metadata.get('history', 'None')}"

           def get_response_schema(self):
               return MY_SCHEMA

           def validate_response(self, response):
               return validate_response(response)
       ```

    4. **Create your prompts** (prompts/agentic/task.jinja):
       ```jinja
       You are analyzing a medical image.
       Image dimensions: {{ width }}x{{ height }} pixels.
       {% if history %}Clinical history: {{ history }}{% endif %}

       Provide your analysis in JSON format with:
       - findings: list of observations
       - diagnosis: primary diagnosis
       - continue: true if you need to use tools, false when done
       ```

Usage:
    ```python
    from nova_retrieval_vlm.nova import NOVAAgenticProcessor

    # Create processor
    processor = NOVAAgenticProcessor(
        model_name="openai/gpt-4o",
        use_tools=True,
        use_web_search=True,
        max_turns=10,
    )

    # Run analysis
    result = await processor.analyze(
        image_path=Path("brain_mri.png"),
        metadata={"history": "Headache for 2 weeks", "modality": "T1-weighted"},
    )

    # Access results
    print(result.final_response["diagnosis"])
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Turns used: {len(result.turns)}")
    ```

See Also:
    - radiant_harness: The underlying VLM agent harness (separate package)
    - nova/processor.py: Full implementation of NOVAAgenticProcessor
    - nova/schemas.py: NOVA JSON schema definition
"""

from __future__ import annotations

from nova_retrieval_vlm.config import NOVAConfig
from nova_retrieval_vlm.config import TaskType
from nova_retrieval_vlm.nova.processor import NOVAAgenticProcessor
from nova_retrieval_vlm.nova.schemas import NOVA_SCHEMA
from nova_retrieval_vlm.nova.schemas import get_required_fields
from nova_retrieval_vlm.nova.schemas import validate_nova_response

__all__ = [
    # Main processor
    "NOVAAgenticProcessor",
    # Configuration
    "NOVAConfig",
    "TaskType",
    # Schema
    "NOVA_SCHEMA",
    "get_required_fields",
    "validate_nova_response",
]
