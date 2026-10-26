"""
LLM client wrapper supporting Groq, Ollama and Google Gemini.
"""

import os
import time
import requests

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


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
        self.model = model or os.getenv("LLM_MODEL", "gemma3:12b")
        self.temperature = temperature
        
        if self.provider == "groq":
            self.model = model or os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
            self.api_key = (os.getenv("GROQ_API_KEY") or "").strip()
            if not self.api_key:
                raise ValueError("GROQ_API_KEY is not set. Add it to Faculty Part/.env")
            self.base_url = GROQ_BASE_URL
            self.ping()
            print(f"✓ Connected to Groq")
            print(f"✓ Using model: {self.model}")
        elif self.provider == "gemini":
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY not set in environment")
            self.base_url = "https://generativelanguage.googleapis.com/v1beta"
            print(f"✓ Using Google Gemini")
            print(f"✓ Model: {self.model}")
        else:
            self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.ping()
            print(f"✓ Connected to Ollama at {self.base_url}")
            print(f"✓ Using model: {self.model}")

    def ping(self) -> None:
        """Raise ConnectionError if the LLM backend is unreachable or the model is unavailable."""
        if self.provider == "gemini":
            return
        if self.provider == "groq":
            try:
                response = requests.get(f"{self.base_url}/models", timeout=10,
                                        headers={"Authorization": f"Bearer {self.api_key}"})
            except requests.exceptions.RequestException as e:
                raise ConnectionError(f"Cannot reach Groq at {self.base_url}. Error: {e}")
            if response.status_code == 401:
                raise ConnectionError("Groq rejected the API key. Check GROQ_API_KEY in Faculty Part/.env")
            response.raise_for_status()
            available = sorted(m.get("id") for m in response.json().get("data", []))
            if self.model not in available:
                raise ConnectionError(
                    f"Groq model '{self.model}' is not available. Set LLM_MODEL to one of: {', '.join(available)}"
                )
            return
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running. Error: {e}"
            )
        pulled = {m.get("name") for m in response.json().get("models", [])}
        if self.model not in pulled and f"{self.model}:latest" not in pulled:
            raise ConnectionError(
                f"Ollama model '{self.model}' is not pulled. Run: ollama pull {self.model}"
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
        if self.provider == "groq":
            return self._generate_groq(prompt, max_tokens, temperature, format, max_retries)
        if self.provider == "gemini":
            return self._generate_gemini(prompt, max_tokens, temperature, format, max_retries)
        return self._generate_ollama(prompt, max_tokens, temperature, format, max_retries)

    def _generate_groq(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        format: str,
        max_retries: int
    ) -> str:
        """Generate text with Groq's OpenAI-compatible chat completions API."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else self.temperature,
            "max_completion_tokens": max_tokens,
        }
        if format == "json":
            payload["response_format"] = {"type": "json_object"}
        if self.model.startswith("openai/gpt-oss"):
            # Reasoning tokens count against max_completion_tokens; keep them short
            payload["reasoning_effort"] = "low"

        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions", json=payload, timeout=120,
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                if response.status_code == 429 and attempt < max_retries:
                    # Free-tier token/request limits: wait as long as Groq asks
                    wait = float(response.headers.get("retry-after") or 2 ** attempt * 5)
                    print(f"[LLM] Groq rate limit, waiting {wait:.0f}s")
                    time.sleep(min(wait, 60))
                    continue
                if response.status_code == 400 and format == "json" and "json_validate_failed" in response.text:
                    # The model produced invalid JSON; Groq returns what it generated
                    failed = response.json().get("error", {}).get("failed_generation")
                    if failed:
                        return failed.strip()
                response.raise_for_status()
                return (response.json()["choices"][0]["message"].get("content") or "").strip() or "{}"
            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    print(f"[LLM] Groq request failed on attempt {attempt + 1}/{max_retries + 1}: {e}, retrying...")
                    time.sleep(2)
                else:
                    detail = getattr(getattr(e, "response", None), "text", "")[:300]
                    raise RuntimeError(f"Groq generation failed: {e} {detail}")
        raise RuntimeError("Groq rate limit still exceeded after retries. Wait a minute and try again.")
    
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
