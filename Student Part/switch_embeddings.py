"""
Quick script to switch embedding models.
"""
import sys
import subprocess
import re

EMBEDDINGS = {
    "1": ("mxbai-embed-large", "MixedBread AI Large - Best accuracy (Recommended)", 1024),
    "2": ("nomic-embed-text", "Nomic Embed Text - Current model", 768),
    "3": ("all-minilm", "All-MiniLM - Fastest, smallest", 384),
}

print("=" * 70)
print("Embedding Model Switcher for NMIMS Academic RAG")
print("=" * 70)

print("\nAvailable embedding models:")
for key, (model, desc, dim) in EMBEDDINGS.items():
    print(f"{key}. {desc}")
    print(f"   Model: {model} ({dim}-dim)")

print("\nCurrent model: nomic-embed-text (768-dim)")

choice = input("\nSelect embedding model (1-3) or 'q' to quit: ").strip()

if choice.lower() == 'q':
    print("Cancelled.")
    sys.exit(0)

if choice not in EMBEDDINGS:
    print("Invalid choice!")
    sys.exit(1)

model, desc, dim = EMBEDDINGS[choice]

print(f"\n{'='*70}")
print(f"Switching to: {model}")
print(f"Description: {desc}")
print(f"Dimensions: {dim}")
print('='*70)

# Check if model is available
print(f"\nChecking if {model} is available...")
result = subprocess.run(
    ["ollama", "list"],
    capture_output=True,
    text=True
)

if model not in result.stdout:
    print(f"\n⚠️  Model not found locally.")
    pull = input(f"Pull {model}? This may take a few minutes. (y/n): ").strip().lower()
    
    if pull == 'y':
        print(f"\nPulling {model}...")
        subprocess.run(["ollama", "pull", model])
        print(f"✅ {model} downloaded!")
    else:
        print("Cancelled. Model not changed.")
        sys.exit(0)
else:
    print(f"✅ {model} is available!")

# Update the files
files_to_update = [
    ("rag/retriever.py", 'OllamaEmbeddings\\(model="[^"]*"\\)'),
    ("ingest/index.py", 'OllamaEmbeddings\\(model="[^"]*"\\)'),
]

print(f"\nUpdating files...")

for filepath, pattern in files_to_update:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        replacement = f'OllamaEmbeddings(model="{model}")'
        new_content = re.sub(pattern, replacement, content)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        
        print(f"✅ Updated {filepath}")
        
    except Exception as e:
        print(f"❌ Error updating {filepath}: {e}")
        sys.exit(1)

print("\n" + "="*70)
print("✅ Embedding model switched successfully!")
print("="*70)

print(f"\nNew embedding model: {model} ({dim}-dim)")

print("\n⚠️  IMPORTANT: You must re-index your documents!")
print("\nNext steps:")
print("1. Re-index documents: python ingest/index.py")
print("   (This will take ~30 seconds)")
print("2. Restart the app: python -m streamlit run app.py")
print("3. Test with a query")

print("\n💡 Why re-index?")
print("   Different embedding models create different vector representations.")
print("   The old vectors won't work with the new model.")
