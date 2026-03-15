# pyright: basic
"""HuggingFace model adapters for the VLM harness.

Provides adapters for local HuggingFace models (text and vision-language).
These are equal-priority alternatives to the OpenAI adapter for local inference.

Note: Requires torch and transformers packages to be installed.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import itertools
import json
import re
from collections import OrderedDict
from collections.abc import AsyncIterator
from io import BytesIO
from typing import TYPE_CHECKING
from typing import Any

from beartype import beartype
from loguru import logger

from radiant_harness.exceptions import ModelError
from radiant_harness.models.adapter_protocol import AdapterProtocol
from radiant_harness.models.adapter_protocol import GenerationLog

if TYPE_CHECKING:
    from PIL import Image
    from transformers import PreTrainedModel  # pyright: ignore[reportMissingImports]
    from transformers import PreTrainedTokenizer  # pyright: ignore[reportMissingImports]

_TOOL_CALL_BLOCK_RE = re.compile(r"```tool\s*(\{.*?\})\s*```", re.DOTALL)

_TOOL_SYSTEM_PREAMBLE = (
    "\n\n# Available Tools\n"
    "You may call tools by emitting a fenced code block with the `tool` tag. "
    "Each block must contain a JSON object with `name` (string) and `arguments` (object).\n\n"
    "Example:\n"
    "```tool\n"
    '{"name": "zoom", "arguments": {"x": 100, "y": 200, "level": 2}}\n'
    "```\n\n"
    "Available tools:\n"
)

_DECODED_IMAGE_CACHE_SIZE = 8


def _format_tools_for_prompt(tools: list[dict[str, Any]]) -> str:
    """Format OpenAI-style tool schemas into text for prompt injection.

    Args:
        tools: List of tool schemas in OpenAI function-calling format.

    Returns:
        Formatted string describing available tools.
    """
    parts: list[str] = [_TOOL_SYSTEM_PREAMBLE]
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        required = set(params.get("required", []))

        parts.append(f"- **{name}**: {desc}")
        if props:
            param_strs: list[str] = []
            for pname, pdef in props.items():
                ptype = pdef.get("type", "any")
                pdesc = pdef.get("description", "")
                req_marker = " (required)" if pname in required else ""
                param_strs.append(f"    - `{pname}` ({ptype}{req_marker}): {pdesc}")
            parts.append("\n".join(param_strs))

    return "\n".join(parts) + "\n"


def _inject_tool_docs(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return a copy of *messages* with tool documentation prepended to the system message.

    If no system message exists, one is inserted at the front. The original
    list is never mutated.
    """
    tool_text = _format_tools_for_prompt(tools)
    messages = [dict(m) for m in messages]  # shallow copy each dict

    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                msg["content"] = content + tool_text
            return messages

    # No system message found — insert one
    messages.insert(0, {"role": "system", "content": tool_text.strip()})
    return messages


_JSON_MODE_INSTRUCTION = (
    "\n\nIMPORTANT: You must respond with a valid JSON object. "
    "Do not include any text before or after the JSON. "
    "Your entire response must be parseable as JSON."
)


def _inject_json_mode(
    messages: list[dict[str, Any]],
    response_format: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Append a JSON-mode instruction to the system message.

    Used when ``response_format`` is requested but the model doesn't natively
    support structured output. When *response_format* contains a
    ``json_schema``, the schema is included in the prompt so the model knows
    the expected output shape.

    The original list is never mutated.
    """
    instruction = _JSON_MODE_INSTRUCTION

    # If a json_schema is provided, include it so the model knows the shape.
    if response_format is not None:
        schema_obj = response_format.get("json_schema", {}).get("schema")
        if schema_obj is not None:
            instruction += (
                "\n\nYou must conform to this JSON schema:\n```json\n"
                f"{json.dumps(schema_obj, indent=2)}\n```"
            )

    messages = [dict(m) for m in messages]

    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                msg["content"] = content + instruction
            return messages

    messages.insert(0, {"role": "system", "content": instruction.strip()})
    return messages


def _require_torch():
    """Import torch, raising ImportError with helpful message if unavailable."""
    try:
        import torch  # pyright: ignore[reportMissingImports]

        return torch
    except ImportError as e:
        raise ImportError(
            "torch is required for HuggingFace adapters. Install with: pip install torch"
        ) from e


def _require_transformers():
    """Import transformers, raising ImportError with helpful message if unavailable."""
    try:
        import transformers  # pyright: ignore[reportMissingImports]

        return transformers
    except ImportError as e:
        raise ImportError(
            "transformers is required for HuggingFace adapters. "
            "Install with: pip install transformers"
        ) from e


class HuggingFaceAdapter(AdapterProtocol):
    """Adapter for HuggingFace text generation models.

    Supports both text-only and vision-language models through AutoModel.
    Provides the same interface as OpenAIAdapter for seamless switching.

    Example:
        adapter = HuggingFaceAdapter(
            model_name="meta-llama/Llama-2-7b-chat-hf",
            device="cuda",
            torch_dtype="float16",
        )

        # Use with processor
        processor = MyProcessor(
            model_name="meta-llama/Llama-2-7b-chat-hf",
            adapter_factory=lambda: adapter,
        )
    """

    supports_multipart_tool_content: bool = False

    @beartype
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_flash_attention: bool = False,
        max_input_length: int | None = None,
    ) -> None:
        """Initialize HuggingFace adapter.

        Args:
            model_name: HuggingFace model identifier or local path
            device: Device to run on ("cuda", "cpu", "mps", or "auto")
            torch_dtype: Data type ("auto", "float16", "bfloat16", "float32")
            trust_remote_code: Whether to trust remote code in model repo
            load_in_8bit: Enable 8-bit quantization (requires bitsandbytes)
            load_in_4bit: Enable 4-bit quantization (requires bitsandbytes)
            use_flash_attention: Enable Flash Attention 2 (requires flash-attn)
            max_input_length: Maximum input token length (None = use model's max)
        """
        self.model_name = model_name
        self._device_str = device
        self._dtype_str = torch_dtype
        self._trust_remote_code = trust_remote_code
        if trust_remote_code:
            logger.warning(
                f"trust_remote_code=True for model {model_name!r}. "
                "This allows arbitrary code execution from the model repository. "
                "Only use this with models you trust."
            )
        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit
        self._use_flash_attention = use_flash_attention
        self._max_input_length = max_input_length

        # Per-instance counter for generating unique tool-call IDs.
        # Avoids global mutable state that would leak IDs across sessions.
        self._tool_call_counter = itertools.count(1)

        # Lazy-loaded components
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizer | None = None
        self._torch: Any = None

    @property
    def torch(self) -> Any:
        """Get torch module (imports on first access)."""
        if self._torch is None:
            self._torch = _require_torch()
        return self._torch

    @property
    def device(self) -> Any:
        """Get the device to run on (torch.device)."""
        torch = self.torch
        if self._device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self._device_str)

    @property
    def dtype(self) -> Any:
        """Get the torch dtype."""
        torch = self.torch
        if self._dtype_str == "auto":
            if torch.cuda.is_available():
                return torch.float16
            return torch.float32
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self._dtype_str, torch.float32)

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get or create the tokenizer."""
        if self._tokenizer is None:
            transformers = _require_transformers()
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self._trust_remote_code,
            )
            # Ensure pad token exists
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    @property
    def model(self) -> PreTrainedModel:
        """Get or create the model."""
        if self._model is None:
            transformers = _require_transformers()

            # Build model kwargs
            model_kwargs: dict[str, Any] = {
                "trust_remote_code": self._trust_remote_code,
            }

            # Quantization options
            if self._load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            elif self._load_in_4bit:
                model_kwargs["load_in_4bit"] = True
            else:
                model_kwargs["torch_dtype"] = self.dtype

            # Flash attention
            if self._use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"

            # Device mapping
            if self._load_in_8bit or self._load_in_4bit or self._device_str == "auto":
                model_kwargs["device_map"] = "auto"

            logger.info(f"Loading HuggingFace model: {self.model_name}")

            self._model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs,
            )

            # Move to device if not using device_map
            if "device_map" not in model_kwargs:
                self._model = self._model.to(self.device)

            self._model.eval()
            logger.info(f"Model loaded on {self.device}")

        return self._model

    @beartype
    async def generate_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        stream: bool = False,
        seed: int | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog] | AsyncIterator[str]:
        """Generate a chat completion using the HuggingFace model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            tools: Tool definitions (handled via prompt engineering)
            response_format: Structured response format (not supported)
            stream: Not supported; raises ModelError if True
            seed: Random seed for reproducibility (sets torch manual seed)

        Returns:
            Tuple of (content, tool_calls, generation_log).
            Streaming is not supported and will raise ModelError.
        """
        if stream:
            raise ModelError(
                "Streaming is not supported for HuggingFace adapters",
                model_name=self.model_name,
            )

        torch = self.torch

        # Inject tool schemas into messages so the model knows what's available
        if tools:
            messages = _inject_tool_docs(messages, tools)

        # HF models don't support response_format natively; emulate via prompt
        if response_format is not None:
            logger.debug("Emulating response_format via JSON-mode system instruction")
            messages = _inject_json_mode(messages, response_format)

        # Convert messages to prompt
        prompt = self._format_messages(messages)

        # Tokenize with memory optimization
        max_input_length = self._max_input_length or self.tokenizer.model_max_length
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        )
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        # Generate with memory optimizations — offload to thread to avoid
        # blocking the event loop during CPU/GPU-bound inference.
        # When temperature <= 0, use greedy decoding (do_sample=False).
        # HF ignores the temperature value when do_sample=False, so we pass
        # 1.0 as a safe placeholder to avoid any division-by-zero in custom
        # samplers.  This matches OpenAI behaviour where temperature=0 means
        # deterministic (greedy) output.
        greedy = temperature <= 0
        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "temperature": 1.0 if greedy else temperature,
            "do_sample": not greedy,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }

        _seed = seed

        def _run_generate():
            if _seed is not None:
                torch.manual_seed(_seed)
            with torch.inference_mode():
                if torch.cuda.is_available():
                    with torch.amp.autocast("cuda"):
                        return self.model.generate(**gen_kwargs)
                return self.model.generate(**gen_kwargs)

        try:
            outputs = await asyncio.to_thread(_run_generate)
        except torch.cuda.OutOfMemoryError as e:
            raise ModelError(f"CUDA out of memory: {e}", model_name=self.model_name) from e
        except RuntimeError as e:
            raise ModelError(
                f"HuggingFace generation failed: {e}", model_name=self.model_name
            ) from e

        # Decode response (only new tokens)
        response_ids = outputs[0][input_length:]
        content = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Parse tool calls if tools were provided
        tool_calls = None
        if tools:
            tool_calls, content = self._parse_tool_calls(content)

        # Determine finish reason: "stop" if EOS was generated, "length" if truncated
        hit_eos = (
            len(response_ids) > 0
            and self.tokenizer.eos_token_id is not None
            and int(response_ids[-1]) == self.tokenizer.eos_token_id
        )
        finish_reason = "stop" if hit_eos else "length"

        # Build generation log
        gen_log = GenerationLog(
            prompt_tokens=input_length,
            completion_tokens=len(response_ids),
            finish_reason=finish_reason,
        )

        logger.debug(f"HuggingFace completion finished, tokens={gen_log.tokens}")

        return content, tool_calls, gen_log

    def _format_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        """Format messages into a prompt string.

        Uses the tokenizer's chat template for message formatting.
        """
        # Use chat template - all modern HF tokenizers support this
        # Convert any multimodal content to text-only for text-only models
        text_messages = self._extract_text_messages(messages)
        return self.tokenizer.apply_chat_template(
            text_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _extract_text_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Extract text-only versions of messages for chat template."""
        text_messages = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [
                    item.get("text", "") for item in content if item.get("type") == "text"
                ]
                content = "\n".join(text_parts)
            text_messages.append(
                {
                    "role": msg.get("role", "user"),
                    "content": content,
                }
            )
        return text_messages

    def _parse_tool_calls(
        self,
        content: str,
    ) -> tuple[list[dict[str, Any]] | None, str]:
        """Parse tool calls from model output.

        Returns:
            Tuple of (tool_calls, remaining_content)
        """
        tool_calls = []

        matches = list(_TOOL_CALL_BLOCK_RE.finditer(content))
        if not matches:
            return None, content

        for i, match in enumerate(matches):
            try:
                call_data = json.loads(match.group(1))
                call_id = f"call_{next(self._tool_call_counter)}"
                tool_calls.append(
                    {
                        "id": call_id,
                        "name": call_data.get("name", ""),
                        "arguments": json.dumps(call_data.get("arguments", {})),
                    }
                )
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed tool call block {i}: {e}")

        # Remove tool blocks from content
        clean_content = _TOOL_CALL_BLOCK_RE.sub("", content).strip()

        return tool_calls if tool_calls else None, clean_content

    async def aclose(self) -> None:
        """Release model and tokenizer to free GPU/CPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._torch = None
        try:
            import torch  # pyright: ignore[reportMissingImports]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class HuggingFaceVLMAdapter(HuggingFaceAdapter):
    """Adapter for HuggingFace Vision-Language Models.

    Extends HuggingFaceAdapter with image processing capabilities for
    models like LLaVA, InstructBLIP, Qwen-VL, etc.

    Example:
        adapter = HuggingFaceVLMAdapter(
            model_name="llava-hf/llava-1.5-7b-hf",
            device="cuda",
        )
    """

    @beartype
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_flash_attention: bool = False,
        max_input_length: int | None = None,
    ) -> None:
        """Initialize HuggingFace VLM adapter."""
        super().__init__(
            model_name=model_name,
            device=device,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            use_flash_attention=use_flash_attention,
            max_input_length=max_input_length,
        )
        self._processor: Any = None
        self._decoded_image_cache: OrderedDict[str, Any] = OrderedDict()

    def _get_cached_image_copy(self, data_url: str) -> Image.Image | None:
        cached = self._decoded_image_cache.get(data_url)
        if cached is None:
            return None
        self._decoded_image_cache.move_to_end(data_url)
        return cached.copy()

    def _store_decoded_image(self, data_url: str, image: Image.Image) -> None:
        self._decoded_image_cache[data_url] = image
        self._decoded_image_cache.move_to_end(data_url)
        while len(self._decoded_image_cache) > _DECODED_IMAGE_CACHE_SIZE:
            _url, evicted = self._decoded_image_cache.popitem(last=False)
            evicted.close()

    @property
    def processor(self):
        """Get or create the processor (tokenizer + image processor)."""
        if self._processor is None:
            transformers = _require_transformers()
            self._processor = transformers.AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=self._trust_remote_code,
            )
        return self._processor

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get tokenizer from processor, ensuring pad_token is set."""
        tok = self.processor.tokenizer
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    @property
    def model(self) -> PreTrainedModel:
        """Get or create the VLM model."""
        if self._model is None:
            transformers = _require_transformers()

            model_kwargs: dict[str, Any] = {
                "trust_remote_code": self._trust_remote_code,
            }

            if self._load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            elif self._load_in_4bit:
                model_kwargs["load_in_4bit"] = True
            else:
                model_kwargs["torch_dtype"] = self.dtype

            if self._use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"

            if self._load_in_8bit or self._load_in_4bit or self._device_str == "auto":
                model_kwargs["device_map"] = "auto"

            logger.info(f"Loading HuggingFace VLM: {self.model_name}")

            # Try Vision2Seq first, then fall back to CausalLM
            try:
                self._model = transformers.AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    **model_kwargs,
                )
                logger.debug(f"Loaded {self.model_name} as Vision2Seq model")
            except (ValueError, OSError, RuntimeError) as e:
                logger.info(
                    f"Vision2Seq not supported for {self.model_name} ({type(e).__name__}), "
                    f"loading as CausalLM"
                )
                self._model = transformers.AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs,
                )

            if "device_map" not in model_kwargs:
                self._model = self._model.to(self.device)

            self._model.eval()
            logger.info(f"VLM loaded on {self.device}")

        return self._model

    @beartype
    async def generate_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        stream: bool = False,
        seed: int | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog] | AsyncIterator[str]:
        """Generate a chat completion with image support.

        Streaming is not supported and will raise ModelError.
        """
        if stream:
            raise ModelError(
                "Streaming is not supported for HuggingFace adapters",
                model_name=self.model_name,
            )

        torch = self.torch

        # Inject tool schemas into messages so the model knows what's available
        if tools:
            messages = _inject_tool_docs(messages, tools)

        # HF models don't support response_format natively; emulate via prompt
        if response_format is not None:
            logger.debug("Emulating response_format via JSON-mode system instruction")
            messages = _inject_json_mode(messages, response_format)

        # Extract images from messages
        images = self._extract_images(messages)

        # Format prompt
        prompt = self._format_messages(messages)

        # Process inputs
        if images:
            inputs = self.processor(
                text=prompt,
                images=images,
                return_tensors="pt",
                padding=True,
            )
        else:
            inputs = self.processor(
                text=prompt,
                return_tensors="pt",
                padding=True,
            )

        inputs = {
            k: v.to(self.device, non_blocking=True) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }
        input_ids = inputs.get("input_ids", inputs.get("inputs_embeds"))
        if input_ids is None:
            raise ModelError(
                "Processor returned neither input_ids nor inputs_embeds",
                model_name=self.model_name,
            )
        input_length = input_ids.shape[1]

        # Generate — offload to thread to avoid blocking the event loop
        # Greedy decoding when temperature <= 0 (parity with OpenAI adapter).
        greedy = temperature <= 0
        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "temperature": 1.0 if greedy else temperature,
            "do_sample": not greedy,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
            "use_cache": True,
        }

        _seed = seed

        def _run_vlm_generate():
            if _seed is not None:
                torch.manual_seed(_seed)
            with torch.inference_mode():
                if torch.cuda.is_available():
                    with torch.amp.autocast("cuda"):
                        return self.model.generate(**gen_kwargs)
                return self.model.generate(**gen_kwargs)

        try:
            outputs = await asyncio.to_thread(_run_vlm_generate)
        except torch.cuda.OutOfMemoryError as e:
            raise ModelError(f"CUDA out of memory in VLM: {e}", model_name=self.model_name) from e
        except RuntimeError as e:
            raise ModelError(
                f"HuggingFace VLM generation failed: {e}", model_name=self.model_name
            ) from e

        # Decode response
        response_ids = outputs[0][input_length:]
        content = self.processor.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Parse tool calls
        tool_calls = None
        if tools:
            tool_calls, content = self._parse_tool_calls(content)

        # Determine finish reason
        eos_id = self.processor.tokenizer.eos_token_id
        hit_eos = len(response_ids) > 0 and eos_id is not None and int(response_ids[-1]) == eos_id
        finish_reason = "stop" if hit_eos else "length"

        gen_log = GenerationLog(
            prompt_tokens=input_length,
            completion_tokens=len(response_ids),
            finish_reason=finish_reason,
        )

        logger.debug(
            f"HuggingFace VLM completion finished, tokens={gen_log.tokens}, images={len(images)}"
        )

        return content, tool_calls, gen_log

    def _extract_images(self, messages: list[dict[str, Any]]) -> list[Image.Image]:
        """Extract PIL images from message content."""
        from PIL import Image as PILImage

        images = []

        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue

            for item in content:
                if item.get("type") != "image_url":
                    continue

                image_url = item.get("image_url", {})
                url = image_url.get("url", "")

                try:
                    if url.startswith("data:"):
                        cached = self._get_cached_image_copy(url)
                        if cached is not None:
                            images.append(cached)
                            continue
                        # Base64 encoded image
                        # Format: data:image/png;base64,<data>
                        if not url.startswith("data:image/"):
                            logger.warning(f"Skipping non-image data URI: {url[:40]}...")
                            continue
                        _, data = url.split(",", 1)
                        image_data = base64.b64decode(data)
                        with PILImage.open(BytesIO(image_data)) as src:
                            image = src.convert("RGB")
                        self._store_decoded_image(url, image.copy())
                    elif url.startswith(("http://", "https://")):
                        # URL - would need httpx to fetch
                        # For now, skip remote URLs
                        logger.warning(f"Remote image URLs not supported: {url[:50]}...")
                        continue
                    else:
                        # Reject arbitrary local file paths to prevent
                        # reading sensitive files via crafted messages.
                        logger.warning(
                            f"Skipping local file path in image_url (not allowed): {url[:80]}..."
                        )
                        continue

                    images.append(image)
                except (OSError, ValueError, binascii.Error) as e:
                    logger.warning(f"Failed to load image: {e}")
                    continue

        return images

    async def aclose(self) -> None:
        """Release model, processor, and image cache to free GPU/CPU memory."""
        # Close cached images before clearing
        for img in self._decoded_image_cache.values():
            img.close()
        self._decoded_image_cache.clear()
        self._processor = None
        await super().aclose()
