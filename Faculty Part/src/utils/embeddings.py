"""
Embedding model wrapper supporting both Sentence Transformers and Ollama.
"""

from typing import List, Union
import os


class EmbeddingModel:
    """
    Wrapper for embedding generation using Sentence Transformers or Ollama.
    """
    
    def __init__(
        self,
        model: str = None,
        use_sentence_transformers: bool = True
    ):
        """
        Initialize embedding model.
        
        Args:
            model: Model name
            use_sentence_transformers: If True, use sentence-transformers; else use Ollama
        """
        self.use_sentence_transformers = use_sentence_transformers
        
        if self.use_sentence_transformers:
            self._init_sentence_transformers(model)
        else:
            self._init_ollama(model)
    
    def _init_sentence_transformers(self, model: str = None):
        """Initialize Sentence Transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Install: pip install sentence-transformers"
            )
        
        self.model_name = model or os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
        
        # Check for GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model
        self.model = SentenceTransformer(self.model_name, device=self.device)
        
        print(f"✓ Initialized embedding model: {self.model_name} on {self.device}")
    
    def _init_ollama(self, model: str = None):
        """Initialize Ollama embedding model."""
        import requests
        
        self.model_name = model or os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Test connection
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            print(f"✓ Initialized embedding model: {self.model_name}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running. Error: {e}"
            )
    
    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text.
        
        Args:
            text: Single text string or list of texts
        
        Returns:
            Single embedding vector or list of vectors
        """
        if self.use_sentence_transformers:
            return self._embed_sentence_transformers(text)
        else:
            return self._embed_ollama(text)
    
    def _embed_sentence_transformers(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using Sentence Transformers."""
        # Handle single text
        if isinstance(text, str):
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        
        # Handle batch
        embeddings = self.model.encode(text, convert_to_numpy=True, show_progress_bar=True)
        return embeddings.tolist()
    
    def _embed_ollama(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using Ollama."""
        import requests
        
        url = f"{self.base_url}/api/embeddings"
        
        # Handle single text
        if isinstance(text, str):
            payload = {
                "model": self.model_name,
                "prompt": text
            }
            try:
                response = requests.post(url, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                return result.get("embedding", [])
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Ollama embedding failed: {e}")
        
        # Handle batch
        embeddings = []
        for t in text:
            payload = {
                "model": self.model_name,
                "prompt": t
            }
            try:
                response = requests.post(url, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                embeddings.append(result.get("embedding", []))
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Ollama embedding failed: {e}")
        
        return embeddings
    
    def get_dimension(self) -> int:
        """
        Get embedding dimension for this model.
        """
        if self.use_sentence_transformers:
            # Get dimension from model
            return self.model.get_sentence_embedding_dimension()
        else:
            # Ollama model dimensions
            dimensions = {
                "llama2": 4096,
                "llama2:7b": 4096,
                "llama2:13b": 5120,
                "llama2:70b": 8192,
                "nomic-embed-text": 768,
                "nomic-embed-text:latest": 768,
            }
            return dimensions.get(self.model_name, 768)

