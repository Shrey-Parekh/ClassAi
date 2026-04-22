# Installation Guide

Complete installation guide for ClassAI project.

## 📋 Prerequisites

### Required Software
- **Python 3.10+** ([Download](https://www.python.org/downloads/))
- **Docker & Docker Compose** ([Download](https://www.docker.com/products/docker-desktop))
- **Ollama** ([Download](https://ollama.ai))
- **Git** ([Download](https://git-scm.com/downloads))

### System Requirements
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 10GB free space
- **CPU**: Multi-core recommended
- **GPU**: Optional (CUDA-capable for faster embeddings)

### Platform-Specific Prerequisites

**Windows:**
```bash
# Install Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Install Tesseract OCR
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

**macOS:**
```bash
brew install tesseract
brew install poppler
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr poppler-utils
sudo apt-get install python3-dev build-essential
```

## 🚀 Installation Steps

### 1. Clone Repository

```bash
git clone <repository-url>
cd ClassAI
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

**For Development:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**For Production:**
```bash
pip install --upgrade pip
pip install -r requirements-prod.txt
```

**For Development with Testing:**
```bash
pip install -r requirements-dev.txt
```

### 4. Install Ollama & Models

```bash
# Install Ollama from https://ollama.ai

# Pull required models
ollama pull gemma3:12b
ollama pull bge-m3

# Verify installation
ollama list
```

### 5. Start Infrastructure

```bash
cd "Faculty Part"
docker-compose up -d

# Verify Qdrant is running
curl http://localhost:6333/health
```

### 6. Configure Environment

```bash
# Copy example environment file
cp "Faculty Part/.env.example" "Faculty Part/.env"
cp "Student Part/.env.example" "Student Part/.env"

# Edit .env files with your settings
# (defaults work for local development)
```

### 7. Index Documents

**Index Faculty Documents:**
```bash
cd "Faculty Part"
python scripts/ingest_new.py
```

**Index Student Documents:**
```bash
cd "Student Part"
python ingest/index.py
```

### 8. Start Application

```bash
cd "Faculty Part"
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 9. Verify Installation

Open browser and navigate to:
- **Homepage**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Qdrant Dashboard**: http://localhost:6334

## 🔧 Configuration

### Environment Variables

Edit `Faculty Part/.env`:

```bash
# LLM Configuration
LLM_PROVIDER=ollama
LLM_MODEL=gemma3:12b
OLLAMA_BASE_URL=http://localhost:11434

# Vector Database
QDRANT_URL=http://localhost:6333
FACULTY_COLLECTION_NAME=faculty_chunks
STUDENT_COLLECTION_NAME=academic_rag

# Embedding Model
EMBEDDING_MODEL=BAAI/bge-m3

# Reranker
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# Application
DEBUG=True
LOG_LEVEL=INFO
```

### GPU Support (Optional)

For CUDA acceleration:

```bash
# Uninstall CPU version
pip uninstall torch

# Install GPU version
pip install torch==2.1.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

## 🧪 Testing Installation

### Run Tests

```bash
cd "Faculty Part"
pytest tests/
```

### Test API

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test query endpoint (requires authentication)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"query": "What is the leave policy?"}'
```

### Test Login

Navigate to http://localhost:8000/signin and login with:
- **Email**: admin@nmims.edu
- **Password**: demo123

## 🐛 Troubleshooting

### Common Issues

**1. Port Already in Use**
```bash
# Check what's using port 8000
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Linux/macOS

# Kill the process or use different port
python -m uvicorn src.api.main:app --port 8001
```

**2. Ollama Not Running**
```bash
# Start Ollama service
ollama serve

# Or restart Ollama application
```

**3. Qdrant Connection Failed**
```bash
# Check Docker containers
docker ps

# Restart Qdrant
cd "Faculty Part"
docker-compose restart
```

**4. Import Errors**
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

**5. CUDA Out of Memory**
```bash
# Reduce batch size or use CPU
export CUDA_VISIBLE_DEVICES=""
```

**6. Permission Denied (Linux/macOS)**
```bash
# Fix permissions
chmod +x scripts/*.py
```

## 📦 Updating

### Update Dependencies

```bash
pip install --upgrade -r requirements.txt
```

### Update Models

```bash
ollama pull gemma3:12b
ollama pull bge-m3
```

### Update Docker Images

```bash
cd "Faculty Part"
docker-compose pull
docker-compose up -d
```

## 🔄 Uninstallation

### Remove Virtual Environment

```bash
deactivate
rm -rf venv  # Linux/macOS
rmdir /s venv  # Windows
```

### Stop Docker Containers

```bash
cd "Faculty Part"
docker-compose down -v
```

### Remove Ollama Models

```bash
ollama rm gemma3:12b
ollama rm bge-m3
```

## 📞 Support

For issues or questions:
1. Check the [README.md](README.md)
2. Review [Troubleshooting](#troubleshooting) section
3. Check existing GitHub issues
4. Create a new issue with:
   - Error message
   - Steps to reproduce
   - System information (OS, Python version)

## 🎓 Next Steps

After successful installation:
1. Read the [README.md](README.md) for usage examples
2. Explore the API documentation at http://localhost:8000/docs
3. Try example queries in the chat interface
4. Review configuration options
5. Set up monitoring and logging for production

---

**Installation Complete!** 🎉

You're ready to use ClassAI. Start by logging in at http://localhost:8000/signin
