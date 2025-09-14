from __future__ import annotations

import asyncio
import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import aiohttp
from loguru import logger
from PIL import Image as PILImage

from .base import BaseAdapter, GenerationLog

API_URL = "https://openrouter.ai/api/v1/chat/completions"

class OpenRouterAdapter(BaseAdapter):
    """Generic adapter for OpenRouter models via the OpenRouter API."""
    def __init__(
        self, 
        model_name: str, 
        api_key: str | None = None,
        max_retries: int = 5,
        timeout: int = 30
    ) -> None:
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY not set")
        super().__init__(key)
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout = timeout
        # Standard headers for OpenRouter API
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": os.getenv("APP_URL", "https://nova-retrieval-vlm.app"),
            "X-Title": os.getenv("APP_NAME", "NOVA Retrieval VLM"),
            "Content-Type": "application/json"
        }
        logger.info(f"Initialized OpenRouterAdapter with model: {model_name}")
        
        # Pre-flight API key authentication check
        self._validate_api_key()
    
    def _validate_api_key(self) -> None:
        """Validate the API key with OpenRouter."""
        try:
            auth_req = Request(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            with urlopen(auth_req, timeout=10) as auth_resp:
                # Parse and validate JSON response
                resp_json = json.load(auth_resp)
                # Expect 'data' key on success, otherwise treat as error
                if 'data' not in resp_json:
                    err = resp_json.get('error', resp_json)
                    raise ValueError(f"OpenRouter API key authentication failed: {err}")
                logger.info("OpenRouter API key validated successfully")
        except HTTPError as e:
            raise ValueError(f"OpenRouter API key authentication failed: HTTP {e.code} - {e.reason}")
        except URLError as e:
            raise ValueError(f"Unable to reach OpenRouter auth endpoint: {e.reason}")
        except Exception as e:
            raise ValueError(f"Error checking OpenRouter API key: {e}")

    async def generate(
        self,
        image_path: Path,
        passages: Sequence[str],
        system_prompt: str | None = None,
    ) -> Tuple[str, GenerationLog]:
        """
        Generate a response by sending image and passages to the OpenRouter model.
        """
        # Debug: entry log
        logger.info(f"OpenRouterAdapter.generate called for model={self.model_name}, image_path={image_path}, passages_count={len(passages)}")
        
        # Load, resize, and compress image
        try:
            pil_img = PILImage.open(image_path).convert("RGB")
            # Resize while maintaining aspect ratio
            max_dimension = 512  # Balance quality and size
            width, height = pil_img.size
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            
            pil_img = pil_img.resize((new_width, new_height))
            
            # Compress image
            buf = BytesIO()
            pil_img.save(buf, format="JPEG", quality=85, optimize=True)
            img_bytes = buf.getvalue()
            # Log image size info
            logger.debug(f"Image size after optimization: {len(img_bytes)} bytes, dimensions: {new_width}x{new_height}")
            img_b64 = base64.b64encode(img_bytes).decode()
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise ValueError(f"Failed to process image at {image_path}: {e}")
        
        # Prepare the prompt
        prompt = system_prompt or "You are an expert neuroradiology assistant."
        
        # Create message with image in content
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Analyze the provided image based on the instructions."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }
        ]
        
        # If there are passages, add them as a separate user message
        if passages:
            passage_text = "\n\n" + "\n---\n".join(passages)
            messages.append({"role": "user", "content": f"Additional context: {passage_text}"})
        
        # Prepare the request payload
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
        }
        
        # Execute request with retry logic
        backoff = 1
        last_exc = None
        
        for attempt in range(self.max_retries):
            logger.info(f"OpenRouterAdapter.generate attempt {attempt+1}/{self.max_retries}")
            try:
                # Use a ClientSession with a reasonable timeout
                timeout_cfg = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
                    async with session.post(
                        API_URL,
                        json=payload,
                        headers=self.headers,
                    ) as resp:
                        status = resp.status
                        logger.info(f"Received HTTP status: {status}")
                        
                        # Check for authentication errors
                        if resp.headers.get("x-clerk-auth-status") == "signed-out":
                            auth_msg = resp.headers.get("x-clerk-auth-message")
                            raise RuntimeError(f"OpenRouter authentication error: {auth_msg}")
                        
                        # Raise for non-200 responses
                        resp.raise_for_status()
                        
                        # Parse JSON response with timeout
                        try:
                            # Use a timeout to avoid hanging on chunked responses
                            response_json = await asyncio.wait_for(resp.json(), timeout=15)
                            
                            # Extract text from the response
                            if not response_json.get("choices"):
                                raise ValueError("No choices in response")
                                
                            text = response_json["choices"][0]["message"]["content"]
                            
                            # Get token usage if available, otherwise estimate
                            if "usage" in response_json:
                                tokens = response_json["usage"]["total_tokens"]
                            else:
                                tokens = len(text.split()) * 2
                            
                            # Estimate cost based on model type
                            is_gpt4 = "gpt-4" in self.model_name.lower()
                            is_claude = "claude" in self.model_name.lower()
                            
                            if is_gpt4:
                                cost = tokens * 0.00003
                            elif is_claude:
                                cost = tokens * 0.00002
                            else:
                                cost = tokens * 0.00001
                                
                            logger.info(f"Generation successful: {tokens} tokens, estimated cost: ${cost:.6f}")
                            return text, GenerationLog(tokens=tokens, cost=cost)
                        except asyncio.TimeoutError:
                            raise RuntimeError("Timed out reading OpenRouter response body")
                        except (json.JSONDecodeError, KeyError) as e:
                            raise ValueError(f"Failed to parse response: {e}")
            except aiohttp.ClientResponseError as e:
                # Handle specific HTTP status codes
                if e.status == 429:  # Rate limit
                    logger.warning(f"Rate limit exceeded on attempt {attempt+1}/{self.max_retries}, retrying in {backoff}s")
                elif e.status == 400:  # Bad request
                    logger.error(f"Bad request error: {e}")
                    raise ValueError(f"Bad request to OpenRouter API: {e}")
                elif e.status >= 500:  # Server errors
                    logger.warning(f"OpenRouter server error on attempt {attempt+1}/{self.max_retries}, retrying in {backoff}s")
                else:
                    logger.warning(f"OpenRouter API returned {e.status} on attempt {attempt+1}/{self.max_retries}, retrying in {backoff}s")
                last_exc = e
            except Exception as e:
                logger.warning(f"OpenRouter API error: {e} on attempt {attempt+1}/{self.max_retries}, retrying in {backoff}s")
                last_exc = e
                
            # Don't sleep on the last attempt
            if attempt < self.max_retries - 1:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)  # Cap backoff at 30 seconds
                
        # If we got here, all retries failed
        raise RuntimeError(f"Failed to generate response after {self.max_retries} retries") from last_exc 

    async def generate_text(
        self,
        prompt_text: str,
        system_prompt: str | None = None,
    ) -> Tuple[str, GenerationLog]:
        """
        Generate a response using only text (no image).
        """
        # Prepare messages
        prompt = system_prompt or "You are an expert neuroradiology assistant."
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": prompt_text},
        ]
        
        # Prepare payload
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
        }
        
        # Execute request with retry logic
        backoff = 1
        last_exc = None
        
        for attempt in range(self.max_retries):
            logger.debug(f"OpenRouterAdapter.generate_text attempt {attempt+1}/{self.max_retries}")
            try:
                # Use a ClientSession with a reasonable timeout
                timeout_cfg = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
                    async with session.post(
                        API_URL,
                        json=payload,
                        headers=self.headers,
                    ) as resp:
                        # Raise for non-200 responses
                        resp.raise_for_status()
                        
                        # Parse JSON response
                        try:
                            response_json = await resp.json()
                            
                            # Extract text from the response
                            if not response_json.get("choices"):
                                raise ValueError("No choices in response")
                                
                            text = response_json["choices"][0]["message"]["content"]
                            
                            # Get token usage if available, otherwise estimate
                            if "usage" in response_json:
                                tokens = response_json["usage"]["total_tokens"]
                            else:
                                tokens = len(text.split()) * 2
                            
                            # Estimate cost based on model type
                            is_gpt4 = "gpt-4" in self.model_name.lower()
                            is_claude = "claude" in self.model_name.lower()
                            
                            if is_gpt4:
                                cost = tokens * 0.00003
                            elif is_claude:
                                cost = tokens * 0.00002
                            else:
                                cost = tokens * 0.00001
                                
                            return text, GenerationLog(tokens=tokens, cost=cost)
                        except (json.JSONDecodeError, KeyError) as e:
                            raise ValueError(f"Failed to parse response: {e}")
            except aiohttp.ClientResponseError as e:
                logger.warning(f"OpenRouter API returned {e.status} on attempt {attempt+1}/{self.max_retries}, retrying in {backoff}s")
                last_exc = e
            except Exception as e:
                logger.warning(f"OpenRouter API error: {e} on attempt {attempt+1}/{self.max_retries}, retrying in {backoff}s")
                last_exc = e
                
            # Don't sleep on the last attempt
            if attempt < self.max_retries - 1:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)  # Cap backoff at 30 seconds
                
        # If we got here, all retries failed
        raise RuntimeError(f"Failed to generate text response after {self.max_retries} retries") from last_exc 