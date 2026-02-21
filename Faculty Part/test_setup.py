"""Quick test to verify setup is correct."""

import sys
from pathlib import Path

def test_imports():
    """Test if all core modules can be imported."""
    errors = []
    
    try:
        from src.chunking.semantic_chunker import SemanticChunker
        print("✓ Chunking module OK")
    except Exception as e:
        errors.append(f"✗ Chunking: {e}")
    
    try:
        from src.retrieval.pipeline import RetrievalPipeline
        print("✓ Retrieval module OK")
    except Exception as e:
        errors.append(f"✗ Retrieval: {e}")
    
    try:
        from src.generation.answer_generator import AnswerGenerator
        print("✓ Generation module OK")
    except Exception as e:
        errors.append(f"✗ Generation: {e}")
    
    try:
        from config.chunking_config import ChunkLevel, IntentType
        print("✓ Config module OK")
    except Exception as e:
        errors.append(f"✗ Config: {e}")
    
    if errors:
        print("\nErrors found:")
        for error in errors:
            print(error)
        return False
    
    print("\n✓ All core modules can be imported")
    return True

def test_ollama():
    """Test if Ollama is running and has Llama 2."""
    import requests
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            if any("llama2" in name for name in model_names):
                print("✓ Ollama is running with Llama 2")
                return True
            else:
                print("✗ Llama 2 not found - run: ollama pull llama2")
                return False
        else:
            print("✗ Ollama not responding correctly")
            return False
    except requests.exceptions.RequestException:
        print("✗ Ollama not running - start with: ollama serve")
        return False

def test_docker():
    """Test if Docker is available."""
    import subprocess
    try:
        result = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✓ Docker is running")
            return True
        else:
            print("✗ Docker command failed")
            return False
    except FileNotFoundError:
        print("✗ Docker not found - install Docker Desktop")
        return False
    except Exception as e:
        print(f"✗ Docker check failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Faculty Part setup...\n")
    
    results = []
    results.append(test_imports())
    results.append(test_ollama())
    results.append(test_docker())
    
    print("\n" + "="*50)
    if all(results):
        print("✓ Setup looks good! Ready to run.")
        print("\nNext steps:")
        print("1. docker-compose up -d")
        print("2. python setup.py")
        print("3. Add documents to data/raw/")
        print("4. python scripts/ingest_documents.py --input data/raw")
    else:
        print("✗ Setup incomplete. Fix the issues above.")
        sys.exit(1)
