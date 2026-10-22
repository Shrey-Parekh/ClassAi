# Student Part

Indexes NMIMS syllabi and question papers into the `academic_rag` Qdrant
collection. Questions are answered by the Faculty Part API, which reads this
collection, so there is no separate query service here.

## Layout

```
Student Part/
├── ingest/
│   ├── extract.py      Reads Markdown files and pulls out course metadata
│   ├── chunker.py      Splits syllabi by course/unit and papers by question
│   └── index.py        Embeds chunks and writes them to Qdrant
├── data/
│   ├── syllabus/
│   └── question_papers/
└── scripts/
    └── reset_db.py     Drops the collection
```

## Indexing

Qdrant (started from `Faculty Part/docker-compose.yml`) and Ollama must be
running, with the embedding model pulled:

```bash
ollama pull bge-m3
cp .env.example .env        # only needed if Qdrant is not on localhost:6333
python ingest/index.py
```

This drops and rebuilds the collection, which takes a few seconds for the
current documents. Restart the Faculty Part API afterwards so it rebuilds its
BM25 index.

To add documents, put the Markdown files in `data/syllabus/` or
`data/question_papers/` and run the indexer again.

## How documents are chunked

- **Syllabi** are split by course, then by unit. Each chunk carries the course
  code, semester, credits and unit number as metadata.
- **Question papers** are split per question, keeping the marks and the
  course outcome (CO) each question maps to.
- After chunking, CO/SO and unit references found in the text are added to
  the metadata so the API can filter on them.

## Options

```
python ingest/index.py --help

  --syllabus-dir   default: data/syllabus
  --qp-dir         default: data/question_papers
  --qdrant-url     default: QDRANT_URL from .env, else http://localhost:6333
  --collection     default: academic_rag
  --append         add to the collection instead of rebuilding it
```

`--append` does not de-duplicate, so only use it for documents that are not
already indexed.
