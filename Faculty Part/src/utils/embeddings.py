"""
Embedding model wrapper for Ollama.
"""

from typing import List, Union
import os
import requests


class EmbeddingModel:
    """
    Wrapper for embedding generation using Ollama.
    """
    
    def __init__(
        self,
        model: str = None,
        base_url: str = None
    ):
        """
        Initialize embedding model for Ollama.
        
        Args:
            model: Model name (default: llama2)
            base_url: Ollama API URL (default: http://localhost:11434)
        """
        self.model = model or os.getenv("EMBEDDING_MODEL", "llama2")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Test connection
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            print(f"✓ Initialized embedding model: {self.model}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running. Error: {e}"
            )
    
    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text using Ollama.
        
        Args:
            text: Single text string or list of texts
        
        Returns:
            Single embedding vector or list of vectors
        """
        url = f"{self.base_url}/api/embeddings"
        
        # Handle single text
        if isinstance(text, str):
            payload = {
                "model": self.model,
                "prompt": text
            }
            try:
                response = requests.post(url, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                return result.get("embedding", [])
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Ollama embedding failed: {e}")
        
        # Handle batch
        embeddings = []
        for t in text:
            payload = {
                "model": self.model,
                "prompt": t
            }
            try:
                response = requests.post(url, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                embeddings.append(result.get("embedding", []))
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Ollama embedding failed: {e}")
        
        return embeddings
    
    def get_dimension(self) -> int:
        """
        Get embedding dimension for this model.
        
        For Llama 2, the dimension is 4096.
        """
        # Llama 2 embedding dimensions
        dimensions = {
            "llama2": 4096,
            "llama2:7b": 4096,
            "llama2:13b": 5120,
            "llama2:70b": 8192,
        }
        return dimensions.get(self.model, 4096)
