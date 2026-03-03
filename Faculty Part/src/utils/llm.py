"""
LLM client wrapper supporting both Ollama and Google Gemini.
"""

import os
from typing import Optional


class LLMClient:
    """
    Wrapper for LLM operations using Ollama or Google Gemini.
    """
    
    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        use_gemini: bool = True,
        temperature: float = 0.1
    ):
        """
        Initialize LLM client.
        
        Args:
            model: Model name
            api_key: API key (for Gemini)
            use_gemini: If True, use Gemini; else use Ollama
            temperature: Sampling temperature
        """
        self.use_gemini = use_gemini
        self.temperature = temperature
        
        if self.use_gemini:
            self._init_gemini(model, api_key)
        else:
            self._init_ollama(model)
    
    def _init_gemini(self, model: str = None, api_key: str = None):
        """Initialize Google Gemini."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai required. Install: pip install google-generativeai"
            )
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=self.api_key)
        
        # Use simple model name - the API will handle the full path
        self.model_name = model or os.getenv("LLM_MODEL", "gemini-pro")
        self.model = genai.GenerativeModel(self.model_name)
        
        print(f"✓ Connected to Google Gemini")
        print(f"✓ Using model: {self.model_name}")
    
    def _init_ollama(self, model: str = None):
        """Initialize Ollama."""
        import requests
        
        self.model_name = model or os.getenv("LLM_MODEL", "llama2")
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Test connection
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            print(f"✓ Connected to Ollama at {self.base_url}")
            print(f"✓ Using model: {self.model_name}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running. Error: {e}"
            )
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = None
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (overrides default)
        
        Returns:
            Generated text
        """
        if self.use_gemini:
            return self._generate_gemini(prompt, max_tokens, temperature)
        else:
            return self._generate_ollama(prompt, max_tokens, temperature)
    
    def _generate_gemini(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = None
    ) -> str:
        """Generate using Gemini."""
        try:
            generation_config = {
                "temperature": temperature or self.temperature,
                "max_output_tokens": max_tokens,
            }
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}")
    
    def _generate_ollama(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = None
    ) -> str:
        """Generate using Ollama."""
        import requests
        
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens,
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama generation failed: {e}")

