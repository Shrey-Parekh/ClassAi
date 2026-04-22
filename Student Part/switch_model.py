"""
Quick script to switch between different LLM models.
"""
import sys
import subprocess

MODELS = {
    "1": ("llama3.1:8b", "Llama 3.1 8B - Best overall (Recommended)"),
    "2": ("mistral:7b", "Mistral 7B - Fastest"),
    "3": ("gemma2:9b", "Gemma 2 9B - Great for academic content"),
    "4": ("phi3:medium", "Phi-3 Medium - Excellent reasoning"),
    "5": ("qwen2.5:14b", "Qwen 2.5 14B - Current model"),
    "6": ("llama3.1:70b", "Llama 3.1 70B - Best quality (requires 64GB+ RAM)"),
}

print("=" * 70)
print("Model Switcher for NMIMS Academic RAG")
print("=" * 70)

print("\nAvailable models:")
for key, (model, desc) in MODELS.items():
    print(f"{key}. {desc}")
    print(f"   Model: {model}")

print("\nCurrent model: qwen2.5:14b")

choice = input("\nSelect model (1-6) or 'q' to quit: ").strip()

if choice.lower() == 'q':
    print("Cancelled.")
    sys.exit(0)

if choice not in MODELS:
    print("Invalid choice!")
    sys.exit(1)

model, desc = MODELS[choice]

print(f"\n{'='*70}")
print(f"Switching to: {model}")
print(f"Description: {desc}")
print('='*70)

# Check if model is available
print(f"\nChecking if {model} is available...")
result = subprocess.run(
    ["ollama", "list"],
    capture_output=True,
    text=True
)

if model.split(':')[0] not in result.stdout:
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

# Update the code
print(f"\nUpdating rag/chain.py...")

try:
    with open("rag/chain.py", "r") as f:
        content = f.read()
    
    # Find and replace the llm_model line
    import re
    pattern = r'llm_model: str = "[^"]*"'
    replacement = f'llm_model: str = "{model}"'
    
    new_content = re.sub(pattern, replacement, content)
    
    with open("rag/chain.py", "w") as f:
        f.write(new_content)
    
    print(f"✅ Updated rag/chain.py")
    
except Exception as e:
    print(f"❌ Error updating file: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ Model switched successfully!")
print("="*70)

print(f"\nNew model: {model}")
print("\nNext steps:")
print("1. Restart the app: python -m streamlit run app.py")
print("2. Test with a query")

print("\nNote: If you also want to upgrade embeddings, see MODEL_RECOMMENDATIONS.md")
