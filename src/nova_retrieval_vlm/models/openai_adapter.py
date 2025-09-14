from __future__ import annotations

import asyncio
import base64
import os
import socket
import time
from io import BytesIO
from pathlib import Path
from typing import Sequence, Tuple, Any

import openai  # Keep this for type hints if needed, but client is OpenAI()
from loguru import logger
from openai import OpenAI  # Import the client class
from PIL import Image as PILImage

from nova_retrieval_vlm.types import APIError
from .base import BaseAdapter, GenerationLog


class OpenAIAdapter(BaseAdapter):
    """Adapter for calling OpenRouter-compatible models via the OpenAI SDK."""
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
        timeout: int = 60,
    ) -> None:
        # Prefer OPENAI_API_KEY over OPENROUTER_API_KEY unless an explicit api_key is provided
        key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY or OPENROUTER_API_KEY not set")
        super().__init__(key)
        # Configure the OpenAI client to point at OpenRouter or other base_url
        self.client = OpenAI(
            api_key=self.api_key, 
            base_url=base_url or "https://openrouter.ai/api/v1",
            timeout=timeout,
            max_retries=max_retries,
            default_headers={
                "HTTP-Referer": os.getenv("APP_URL", "https://nova-retrieval-vlm.app"),
                "X-Title": os.getenv("APP_NAME", "NOVA Retrieval VLM"),
            }
        )
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout = timeout
        logger.info(f"Initialized OpenAIAdapter with model: {model_name}")

    async def generate(
        self,
        image_path: Path,
        passages: Sequence[str],
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Tuple[str, GenerationLog]:
        # Load and optimize image
        try:
            pil_img = PILImage.open(image_path).convert("RGB")
            # Keep the ORIGINAL resolution – do not downscale.  We still
            # compress to JPEG for efficient transfer.
            
            # Compress image
            buf = BytesIO()
            pil_img.save(buf, format="JPEG", quality=85, optimize=True)
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            logger.debug(f"Image size after compression: {len(buf.getvalue())} bytes")
        except (OSError, IOError) as e:
            logger.error(f"Failed to read image file: {e}")
            raise APIError(f"Failed to read image at {image_path}: {e}") from e
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise APIError(f"Failed to process image at {image_path}: {e}") from e

        # system_prompt (from load_prompt) contains all textual instructions, role, and passages.
        if not system_prompt:
            # Fallback, though cli.py should always provide a comprehensive prompt.
            system_prompt_text = "You are an expert neuroradiology assistant. Please analyze the provided image."
        else:
            system_prompt_text = system_prompt

        # User message content will be a list: one part for text, one for image.
        user_message_content_parts = [
            {
                "type": "text",
                "text": "Analyze the provided image based on the instructions." # Brief contextual text
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}"
                }
            }
        ]

        messages = [
            {"role": "system", "content": system_prompt_text},
            {"role": "user", "content": user_message_content_parts},
        ]
        
        # Call OpenAI SDK with proper error handling
        for attempt in range(self.max_retries):
            try:
                def _call() -> Any:
                    kwargs = {
                        "model": self.model_name,
                        "messages": messages,
                        "timeout": self.timeout,
                    }
                    if max_tokens is not None:
                        kwargs["max_tokens"] = max_tokens
                    if temperature is not None:
                        kwargs["temperature"] = temperature
                    return self.client.chat.completions.create(**kwargs)
                resp = await asyncio.to_thread(_call)
                text = resp.choices[0].message.content
                # Calculate approximate token usage
                tokens = resp.usage.total_tokens if hasattr(resp, 'usage') and resp.usage else len(text.split()) * 2
                # Better cost estimate based on model name (simplified)
                is_gpt4 = "gpt-4" in self.model_name.lower()
                is_claude = "claude" in self.model_name.lower()
                
                # Very rough cost estimation
                if is_gpt4:
                    cost = tokens * 0.00003
                elif is_claude:
                    cost = tokens * 0.00002
                else:
                    cost = tokens * 0.00001
                    
                return text, GenerationLog(
                    model_name=self.model_name,
                    tokens=tokens, 
                    cost=cost,
                    timestamp=time.time()
                )
            except openai.RateLimitError as e:
                logger.warning(f"Rate limit exceeded on attempt {attempt+1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    backoff = 2 ** attempt + 1
                    await asyncio.sleep(backoff)
                else:
                    raise RuntimeError(f"Rate limit exceeded after {self.max_retries} attempts") from e
            except (openai.APIError, openai.APIConnectionError, socket.timeout) as e:
                logger.warning(f"API error on attempt {attempt+1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    backoff = 2 ** attempt + 1
                    await asyncio.sleep(backoff)
                else:
                    raise RuntimeError(f"API error after {self.max_retries} attempts") from e
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise APIError(f"Failed to generate response: {e}") from e
        
        # This should never be reached, but needed for type checking
        raise RuntimeError("All retry attempts exhausted")

    async def generate_text(
        self,
        prompt_text: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Tuple[str, GenerationLog]:
        prompt = system_prompt or "You are an expert neuroradiology assistant."
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": prompt_text},
        ]
        
        # Call OpenAI SDK with proper error handling
        for attempt in range(self.max_retries):
            try:
                def _call() -> Any:
                    kwargs = {
                        "model": self.model_name,
                        "messages": messages,
                        "timeout": self.timeout,
                    }
                    if max_tokens is not None:
                        kwargs["max_tokens"] = max_tokens
                    if temperature is not None:
                        kwargs["temperature"] = temperature
                    return self.client.chat.completions.create(**kwargs)
                resp = await asyncio.to_thread(_call)
                text = resp.choices[0].message.content
                tokens = resp.usage.total_tokens if hasattr(resp, 'usage') and resp.usage else len(text.split()) * 2
                # Better cost estimate based on model name (simplified)
                is_gpt4 = "gpt-4" in self.model_name.lower()
                is_claude = "claude" in self.model_name.lower()
                
                if is_gpt4:
                    cost = tokens * 0.00003
                elif is_claude:
                    cost = tokens * 0.00002
                else:
                    cost = tokens * 0.00001
                    
                return text, GenerationLog(
                    model_name=self.model_name,
                    tokens=tokens, 
                    cost=cost,
                    timestamp=time.time()
                )
            except openai.RateLimitError as e:
                logger.warning(f"Rate limit exceeded on attempt {attempt+1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    backoff = 2 ** attempt + 1
                    await asyncio.sleep(backoff)
                else:
                    raise RuntimeError(f"Rate limit exceeded after {self.max_retries} attempts") from e
            except (openai.APIError, openai.APIConnectionError, socket.timeout) as e:
                logger.warning(f"API error on attempt {attempt+1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    backoff = 2 ** attempt + 1
                    await asyncio.sleep(backoff)
                else:
                    raise RuntimeError(f"API error after {self.max_retries} attempts") from e
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise APIError(f"Failed to generate text response: {e}") from e 