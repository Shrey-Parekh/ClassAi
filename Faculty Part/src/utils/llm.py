"""
LLM client wrapper supporting Ollama and Google Gemini.
"""

import os
import requests
from typing import Optional


class LLMClient:
    """
    Wrapper for LLM operations supporting multiple providers.
    """
    
    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        temperature: float = 0.1
    ):
        """
        Initialize LLM client.
        
        Args:
            model: Model name
            base_url: API URL (for Ollama)
            temperature: Sampling temperature
        """
        self.provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        self.model = model or os.getenv("LLM_MODEL", "llama2")
        self.temperature = temperature
        
        if self.provider == "gemini":
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY not set in environment")
            self.base_url = "https://generativelanguage.googleapis.com/v1beta"
            print(f"✓ Using Google Gemini")
            print(f"✓ Model: {self.model}")
        else:
            self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            # Test Ollama connection
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
        format: str = None,
        max_retries: int = 2
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            format: Output format constraint ("json" for JSON output)
            max_retries: Number of retries on failure
        
        Returns:
            Generated text
        """
        if self.provider == "gemini":
            return self._generate_gemini(prompt, max_tokens, temperature, format, max_retries)
        else:
            return self._generate_ollama(prompt, max_tokens, temperature, format, max_retries)
    
    def _generate_gemini(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        format: str,
        max_retries: int
    ) -> str:
        """Generate text using Google Gemini API."""
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        
        temp = temperature if temperature is not None else self.temperature
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": temp,
                "maxOutputTokens": max_tokens,
                "topP": 0.9,
                "topK": 40,
            }
        }
        
        # Add JSON mode if requested
        if format == "json":
            payload["generationConfig"]["responseMimeType"] = "application/json"
        
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(url, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                
                # Extract text from Gemini response
                if "candidates" in result and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        generated = candidate["content"]["parts"][0].get("text", "").strip()
                        
                        if not generated or generated == "{}":
                            if attempt < max_retries:
                                print(f"[LLM] Empty response on attempt {attempt + 1}/{max_retries + 1}, retrying...")
                                import time
                                time.sleep(2)
                                continue
                            else:
                                print(f"[LLM] All retries exhausted, returning empty response")
                                return "{}"
                        
                        return generated
                
                # No valid response
                if attempt < max_retries:
                    print(f"[LLM] Invalid response structure on attempt {attempt + 1}/{max_retries + 1}, retrying...")
                    import time
                    time.sleep(2)
                else:
                    return "{}"
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    # Rate limit - use exponential backoff
                    wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s
                    if attempt < max_retries:
                        print(f"[LLM] Rate limit hit on attempt {attempt + 1}/{max_retries + 1}, waiting {wait_time}s...")
                        import time
                        time.sleep(wait_time)
                    else:
                        raise RuntimeError(f"Gemini rate limit exceeded after {max_retries + 1} attempts. Please wait a minute and try again.")
                else:
                    if attempt < max_retries:
                        print(f"[LLM] HTTP error on attempt {attempt + 1}/{max_retries + 1}: {e}, retrying...")
                        import time
                        time.sleep(2)
                    else:
                        raise RuntimeError(f"Gemini generation failed: {e}")
                        
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    print(f"[LLM] Timeout on attempt {attempt + 1}/{max_retries + 1}, retrying...")
                    import time
                    time.sleep(2)
                else:
                    raise RuntimeError(f"Gemini generation timed out after {max_retries + 1} attempts")
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    print(f"[LLM] Request error on attempt {attempt + 1}/{max_retries + 1}: {e}, retrying...")
                    import time
                    time.sleep(2)
                else:
                    raise RuntimeError(f"Gemini generation failed: {e}")
        
        return "{}"
    
    def _generate_ollama(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        format: str,
        max_retries: int
    ) -> str:
        """Generate text using Ollama API."""
        url = f"{self.base_url}/api/generate"
        
        temp = temperature if temperature is not None else self.temperature
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": max_tokens,
                "num_ctx": 8192,
                "num_gpu": -1,
                "num_thread": 8,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
            }
        }
        
        if format:
            payload["format"] = format
        
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(url, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                generated = result.get("response", "").strip()
                
                if not generated or generated == "{}":
                    if attempt < max_retries:
                        print(f"[LLM] Empty response on attempt {attempt + 1}/{max_retries + 1}, retrying...")
                        import time
                        time.sleep(2)
                        continue
                    else:
                        print(f"[LLM] All retries exhausted, returning empty response")
                        return "{}"
                
                return generated
                
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    print(f"[LLM] Timeout on attempt {attempt + 1}/{max_retries + 1}, retrying...")
                    import time
                    time.sleep(2)
                else:
                    raise RuntimeError(f"Ollama generation timed out after {max_retries + 1} attempts")
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    print(f"[LLM] Request error on attempt {attempt + 1}/{max_retries + 1}: {e}, retrying...")
                    import time
                    time.sleep(2)
                else:
                    raise RuntimeError(f"Ollama generation failed: {e}")
        
        return "{}"
