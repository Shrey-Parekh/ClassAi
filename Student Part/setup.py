"""
Automated setup script for CLASS AI Student RAG system.
Checks prerequisites and guides through setup process.
"""
import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_step(step_num, text):
    """Print step number."""
    print(f"\n[Step {step_num}] {text}")


def run_command(cmd, description, check=True):
    """Run shell command with description."""
    print(f"  Running: {description}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=check
        )
        if result.returncode == 0:
            print(f"  ✅ {description} - Success")
            return True
        else:
            print(f"  ❌ {description} - Failed")
            if result.stderr:
                print(f"     Error: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"  ❌ {description} - Error: {e}")
        return False


def check_python_version():
    """Check if Python version is adequate."""
    print_step(1, "Checking Python Version")
    
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor} detected")
        print("     Python 3.9+ required")
        return False


def check_ollama():
    """Check if Ollama is installed."""
    print_step(2, "Checking Ollama Installation")
    
    result = subprocess.run(
        "ollama --version",
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"  ✅ Ollama installed: {result.stdout.strip()}")
        return True
    else:
        print("  ❌ Ollama not found")
        print("     Install from: https://ollama.com")
        return False


def check_ollama_running():
    """Check if Ollama server is running."""
    print_step(3, "Checking Ollama Server")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("  ✅ Ollama server is running")
            return True
    except:
        pass
    
    print("  ⚠️  Ollama server not running")
    print("     Start with: ollama serve")
    return False


def pull_ollama_models():
    """Pull required Ollama models."""
    print_step(4, "Pulling Ollama Models")
    
    models = [
        ("qwen2.5:14b", "LLM Model"),
        ("nomic-embed-text", "Embedding Model")
    ]
    
    all_success = True
    for model, description in models:
        print(f"\n  Pulling {description} ({model})...")
        print("  This may take several minutes on first run...")
        
        success = run_command(
            f"ollama pull {model}",
            f"Pull {model}",
            check=False
        )
        
        if not success:
            all_success = False
            print(f"     Try manually: ollama pull {model}")
    
    return all_success


def create_virtual_environment():
    """Create Python virtual environment."""
    print_step(5, "Creating Virtual Environment")
    
    if os.path.exists("venv"):
        print("  ✅ Virtual environment already exists")
        return True
    
    return run_command(
        f"{sys.executable} -m venv venv",
        "Create virtual environment"
    )


def install_requirements():
    """Install Python requirements."""
    print_step(6, "Installing Python Packages")
    
    # Determine pip path
    if sys.platform == "win32":
        pip_path = "venv\\Scripts\\pip"
    else:
        pip_path = "venv/bin/pip"
    
    if not os.path.exists(pip_path):
        pip_path = "pip"  # Fallback to system pip
    
    return run_command(
        f"{pip_path} install -r requirements.txt",
        "Install requirements"
    )


def create_data_directories():
    """Create data directories if they don't exist."""
    print_step(7, "Creating Data Directories")
    
    dirs = [
        "data/syllabus",
        "data/question_papers",
        "data/raw/syllabus",
        "data/raw/question_papers"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("  ✅ Data directories created")
    return True


def check_sample_files():
    """Check if sample files exist."""
    print_step(8, "Checking Sample Files")
    
    sample_files = [
        "data/syllabus/sample_machine_learning.md",
        "data/syllabus/sample_cyber_security.md",
        "data/question_papers/sample_ml_final_2024.md"
    ]
    
    found = sum(1 for f in sample_files if os.path.exists(f))
    
    if found > 0:
        print(f"  ✅ Found {found} sample files")
        return True
    else:
        print("  ⚠️  No sample files found")
        print("     Sample files should be in data/syllabus and data/question_papers")
        return False


def index_documents():
    """Run document indexing."""
    print_step(9, "Indexing Documents")
    
    print("  This will create the vector database...")
    print("  It may take a few minutes depending on document count...")
    
    # Determine python path
    if sys.platform == "win32":
        python_path = "venv\\Scripts\\python"
    else:
        python_path = "venv/bin/python"
    
    if not os.path.exists(python_path):
        python_path = sys.executable
    
    return run_command(
        f"{python_path} ingest/index_documents.py",
        "Index documents",
        check=False
    )


def run_tests():
    """Run system tests."""
    print_step(10, "Running System Tests")
    
    # Determine python path
    if sys.platform == "win32":
        python_path = "venv\\Scripts\\python"
    else:
        python_path = "venv/bin/python"
    
    if not os.path.exists(python_path):
        python_path = sys.executable
    
    return run_command(
        f"{python_path} test_system.py",
        "Run system tests",
        check=False
    )


def main():
    """Main setup process."""
    print_header("CLASS AI Student - Automated Setup")
    print("\nThis script will guide you through the setup process.")
    print("Press Ctrl+C at any time to cancel.")
    
    try:
        input("\nPress Enter to continue...")
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        return 1
    
    # Run setup steps
    steps = [
        ("Python Version", check_python_version, True),
        ("Ollama Installation", check_ollama, True),
        ("Ollama Server", check_ollama_running, False),
        ("Ollama Models", pull_ollama_models, False),
        ("Virtual Environment", create_virtual_environment, True),
        ("Python Packages", install_requirements, True),
        ("Data Directories", create_data_directories, True),
        ("Sample Files", check_sample_files, False),
        ("Document Indexing", index_documents, False),
        ("System Tests", run_tests, False),
    ]
    
    results = {}
    for step_name, step_func, required in steps:
        try:
            success = step_func()
            results[step_name] = success
            
            if required and not success:
                print(f"\n❌ Required step '{step_name}' failed!")
                print("   Please fix the issue and run setup again.")
                return 1
                
        except KeyboardInterrupt:
            print("\n\nSetup cancelled.")
            return 1
        except Exception as e:
            print(f"\n❌ Error in step '{step_name}': {e}")
            results[step_name] = False
            if required:
                return 1
    
    # Summary
    print_header("Setup Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nCompleted: {passed}/{total} steps")
    
    for step_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {step_name}")
    
    if passed == total:
        print("\n" + "=" * 70)
        print("  🎉 Setup Complete! Your system is ready to use.")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Activate virtual environment:")
        if sys.platform == "win32":
            print("     venv\\Scripts\\activate")
        else:
            print("     source venv/bin/activate")
        print("\n  2. Start the application:")
        print("     streamlit run app.py")
        print("\n  3. Open browser:")
        print("     http://localhost:8501")
        return 0
    else:
        print("\n" + "=" * 70)
        print("  ⚠️  Setup completed with some warnings")
        print("=" * 70)
        print("\nYou can still try running the application:")
        print("  streamlit run app.py")
        print("\nOr fix the issues and run setup again:")
        print("  python setup.py")
        return 0


if __name__ == "__main__":
    sys.exit(main())
