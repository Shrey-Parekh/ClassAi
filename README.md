# ClassAI

ClassAI is a question-answering assistant for NMIMS. Faculty can ask about
institutional policies and guidelines, and students can ask about syllabi and
past question papers. Every answer is grounded in the source documents and
cites them.

It runs fully on a local machine. Qdrant stores the vectors, Ollama serves the
language model, and a FastAPI backend handles retrieval, generation, sign-in
and the web UI.

## How it works

1. **Ingestion.** Faculty PDFs and student Markdown files are chunked,
   embedded with `BAAI/bge-m3`, and stored in two Qdrant collections:
   `faculty_chunks` and `academic_rag`.
2. **Routing.** Each query is normalised, abbreviations such as *ML* and *CS*
   are expanded, and its intent is detected. The user's role decides which
   collections it may search.
3. **Retrieval.** Dense vector search and BM25 run in parallel. Results are
   merged with reciprocal rank fusion and then reranked with
   `BAAI/bge-reranker-v2-m3`.
4. **Generation.** The top chunks go to the LLM (`gemma3:12b` via Ollama by
   default). The LLM returns a structured answer with sources, streamed to the
   browser over Server-Sent Events.

### Roles

| Role    | Can search                                        |
|---------|---------------------------------------------------|
| Student | Student collection only                           |
| Faculty | Faculty, student, or both, chosen in the UI       |
| Admin   | Everything                                        |

## Repository layout

```
ClassAI/
├── Faculty Part/            Main application: API, retrieval, web UI
│   ├── src/
│   │   ├── api/             FastAPI app and routes
│   │   ├── chunking/        Structure-aware document chunker
│   │   ├── ingestion/       PDF processing pipeline
│   │   ├── retrieval/       Hybrid search, reranker, scope router
│   │   ├── generation/      Prompting and answer formatting
│   │   └── utils/           Embeddings, Qdrant client, cache, rate limiting
│   ├── frontend/            Sign-in and chat pages (plain HTML/CSS/JS)
│   ├── config/              Chunking and retrieval settings
│   ├── scripts/             Ingestion and maintenance scripts
│   ├── eval/                Golden queries and retrieval metrics
│   ├── tests/
│   └── docker-compose.yml   Qdrant
│
├── Student Part/            Ingestion for syllabi and question papers
│   ├── ingest/              Markdown extraction, chunking, indexing
│   ├── data/
│   │   ├── syllabus/
│   │   └── question_papers/
│   └── scripts/
│
├── requirements.txt         Shared dependencies for both parts
├── requirements-dev.txt
├── requirements-prod.txt
└── INSTALL.md               Detailed setup and troubleshooting
```

## Getting started

### Requirements

- Python 3.10 or newer
- Docker, for Qdrant
- [Ollama](https://ollama.com)
- About 16 GB of RAM. A CUDA GPU is optional but makes embedding much faster.

### 1. Install dependencies

```bash
git clone https://github.com/Shrey-Parekh/ClassAi.git
cd ClassAi
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start Qdrant and pull the models

```bash
cd "Faculty Part"
docker compose up -d

ollama pull gemma3:12b
ollama pull bge-m3
```

### 3. Configure

```bash
cp "Faculty Part/.env.example" "Faculty Part/.env"
cp "Student Part/.env.example" "Student Part/.env"
```

The defaults point at a local Qdrant and Ollama. The one thing you must set is
the sign-in accounts in `Faculty Part/.env`. Each account uses the format
`email:bcrypt_hash:role:name`. To generate a hash:

```bash
python -c "import bcrypt; print(bcrypt.hashpw(b'your-password', bcrypt.gensalt()).decode())"
```

### 4. Index the documents

Faculty documents are PDFs in `Faculty Part/data/raw/`. They are not tracked
in git. Describe each file in `data/metadata.json`, using
`metadata.example.json` as the template.

```bash
cd "Faculty Part"
python scripts/ingest_new.py --input data/raw --metadata data/metadata.json
```

Student documents are Markdown files under `Student Part/data/`:

```bash
cd "Student Part"
python ingest/index.py            # add --append to keep the existing collection
```

### 5. Run

```bash
cd "Faculty Part"
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000/signin>. Interactive API docs are at `/docs`.

## API

| Method   | Path                         | Purpose                                 |
|----------|------------------------------|-----------------------------------------|
| `POST`   | `/api/auth/signin`           | Exchange email and password for a token |
| `POST`   | `/query`                     | Ask a question (streaming or JSON)      |
| `GET`    | `/health`                    | Service and dependency status           |
| `POST`   | `/conversation/new`          | Start a conversation                    |
| `GET`    | `/conversation/{session_id}` | Fetch a conversation's history          |
| `DELETE` | `/conversation/{session_id}` | Delete a conversation                   |
| `GET`    | `/conversations`             | List conversations                      |

Example query:

```bash
curl -N http://localhost:8000/query \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "List all units in Machine Learning", "scope": "student", "stream": true}'
```

## Configuration

Runtime settings live in `Faculty Part/.env`:

| Variable                  | Default                  | Notes                         |
|---------------------------|--------------------------|-------------------------------|
| `LLM_PROVIDER`            | `ollama`                 | `ollama` or `gemini`          |
| `LLM_MODEL`               | `gemma3:12b`             |                               |
| `OLLAMA_BASE_URL`         | `http://localhost:11434` |                               |
| `GEMINI_API_KEY`          |                          | Only when using Gemini        |
| `QDRANT_URL`              | `http://localhost:6333`  |                               |
| `QDRANT_API_KEY`          |                          | Only for a secured Qdrant     |
| `FACULTY_COLLECTION_NAME` | `faculty_chunks`         |                               |
| `STUDENT_COLLECTION_NAME` | `academic_rag`           |                               |
| `DEMO_USER_*`             |                          | Sign-in accounts, see above   |

Chunk sizes and per-intent retrieval limits are in
`Faculty Part/config/chunking_config.py`. Course abbreviations are in
`Faculty Part/src/retrieval/scope_router.py`.

## Tests and evaluation

```bash
cd "Faculty Part"
pytest tests/
```

`eval/golden/queries.jsonl` holds a set of reference questions.
`eval/metrics.py` has the scoring functions: recall@k, MRR, and checks that
answers are grounded in the retrieved text.

## Known limitations

This is not ready for public deployment yet. Specifically:

- Accounts are read from environment variables, and session tokens are kept
  in memory, so restarting the server signs everyone out.
- There is no HTTPS, token expiry, or audit logging.
- Rate limiting is per IP, not per user.

See [INSTALL.md](INSTALL.md) for platform-specific setup and troubleshooting.
