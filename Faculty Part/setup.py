"""
Setup script to initialize the Faculty Part RAG system.
"""

from pathlib import Path
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create necessary directories
DIRS = [
    "data/raw",
    "data/processed",
    "logs",
    "qdrant_storage",
]

def create_directories():
    """Create necessary directories."""
    for dir_path in DIRS:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def check_environment():
    """Check if Ollama is accessible."""
    import requests
    
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            print(f"✓ Ollama is running at {ollama_url}")
            
            # Check if llama2 is available
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            if any("llama2" in name for name in model_names):
                print("✓ Llama 2 model is available")
            else:
                print("⚠ Llama 2 not found. Run: ollama pull llama2")
                return False
            
            return True
        else:
            print(f"⚠ Ollama responded with status {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("⚠ Cannot connect to Ollama")
        print("  Make sure Ollama is running: ollama serve")
        print("  And Llama 2 is pulled: ollama pull llama2")
        return False

def main():
    print("Setting up Faculty Part RAG system...\n")
    
    create_directories()
    print()
    
    if check_environment():
        print("\n✓ Setup complete!")
        print("\nNext steps:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Start Qdrant: docker-compose up -d")
        print("3. Place documents in data/raw/")
        print("4. Run ingestion: python scripts/ingest_documents.py --input data/raw")
        print("5. Start API: python -m src.api.main")
    else:
        print("\n⚠ Setup incomplete. Please configure Ollama.")

if __name__ == "__main__":
    main()
