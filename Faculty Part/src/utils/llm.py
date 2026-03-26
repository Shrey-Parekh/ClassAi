"""
LLM client wrapper for Ollama.
"""

import os
import requests
from typing import Optional


class LLMClient:
    """
    Wrapper for LLM operations using Ollama.
    """
    
    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        temperature: float = 0.1
    ):
        """
        Initialize LLM client for Ollama.
        
        Args:
            model: Model name (default: llama2)
            base_url: Ollama API URL (default: http://localhost:11434)
            temperature: Sampling temperature
        """
        self.model = model or os.getenv("LLM_MODEL", "llama2")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.temperature = temperature
        
        # Test connection
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            print(f"✓ Connected to Ollama at {self.base_url}")
            print(f"✓ Using model: {self.model}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running. Error: {e}"
            )
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = None,
        format: str = None
    ) -> str:
        """
        Generate text from prompt using Ollama.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (default: 4096, increased for detailed responses)
            temperature: Sampling temperature (overrides default, recommended: 0.2 for factual responses)
            format: Output format constraint ("json" forces valid JSON output)
        
        Returns:
            Generated text
        """
        url = f"{self.base_url}/api/generate"
        
        # Use provided temperature or default
        temp = temperature if temperature is not None else self.temperature
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": max_tokens,
                "num_ctx": 16384,  # 16K context window
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,  # Prevents repetition in long outputs
            }
        }
        
        # Add format constraint if specified (forces JSON output at token sampling level)
        if format:
            payload["format"] = format
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama generation failed: {e}")


