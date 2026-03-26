# Implementation Plan: 13 Critical Fixes

## Status: In Progress

This document tracks the implementation of all 13 recommended fixes.

---

## ✅ Completed Fixes

### Fix 4: Title Stripping Inconsistency
**Status:** IMPLEMENTED

**Changes:**
- Created `src/utils/query_normalizer.py` - Single source of truth for text normalization
- `QueryNormalizer.normalize_query()` handles title stripping, whitespace, lowercase
- Centralized logic eliminates scattered inline stripping

**Next Steps:**
- Update `query_understanding.py` to use `QueryNormalizer`
- Update `retrieval/pipeline.py` to use `QueryNormalizer`
- Update `hybrid_search.py` to use `QueryNormalizer`

---

### Fix 6: BM25 Tokenization
**Status:** IMPLEMENTED

**Changes:**
- Created `src/utils/bm25_tokenizer.py` with NLTK word_tokenize
- Domain-specific stopwords: nmims, university, college, document, section
- Standard English stopwords removed
- Fallback to regex tokenizer if NLTK unavailable
- Added `nltk==3.8.1` to requirements.txt

**Next Steps:**
- Update `hybrid_search.py` to use `BM25Tokenizer`
- Rebuild BM25 index after deployment

---

### Fix 13: Evaluation Baseline
**Status:** IMPLEMENTED

**Changes:**
- Created `tests/eval_baseline.py` with 7 test cases
- Categories: faculty_lookup, faculty_research, policy_lookup, procedure, eligibility, form_lookup, general
- Automated pass/fail scoring
- JSON output with timestamp
- Run before/after changes to measure impact

**Usage:**
```bash
python tests/eval_baseline.py
```

---

## 🚧 In Progress

### Fix 1: Two Competing Chunking Strategies
**Status:** PARTIAL

**Changes Made:**
- Added `chunk_level` field to `Chunk` dataclass
- Added `_assign_chunk_levels()` method to `document_chunker.py`
- Levels assigned based on position and content: overview, section, atomic

**Remaining Work:**
- Delete `src/chunking/semantic_chunker.py` entirely
- Update `ingestion/pipeline.py` to use only `document_chunker.py`
- Update all imports to remove semantic_chunker references
- Test ingestion pipeline with new unified strategy

---

## 📋 Pending Fixes

### Fix 2: Arbitrary Token Limits Breaking Meaning
**Priority:** HIGH

**Implementation Plan:**
1. Update `_split_by_size()` in `document_chunker.py`:
   - Walk backward to sentence boundary (`. ` + capital letter)
   - Never split inside numbered lists (detect `\n\d+\.` patterns)
   - Keep lists whole even if oversized
2. Update `chunk_preprocessor.py`:
   - Keep 1960 char hard cap as safety valve
   - Should rarely trigger after sentence-boundary splitting

**Files to Modify:**
- `src/chunking/document_chunker.py`
- `src/utils/chunk_preprocessor.py`

---

### Fix 3: No Citation Tracking
**Priority:** HIGH

**Implementation Plan:**
1. Update `answer_generator.py`:
   - Number chunks in prompt: [1], [2], [3]
   - Add instruction to LLM: "Annotate every factual claim with [N]"
   - Parse output to extract citation numbers
2. Update `prompt_templates.py`:
   - Add citation instruction to all prompts
   - Show example with inline citations
3. Update frontend `chat.js`:
   - Render [N] as superscripts
   - Link to source metadata on click

**Files to Modify:**
- `src/generation/answer_generator.py`
- `src/generation/prompt_templates.py`
- `frontend/chat.js`
- `frontend/chat.css`

---

### Fix 5: Single-Shot Retrieval
**Priority:** MEDIUM

**Implementation Plan:**
1. Update `retrieval/pipeline.py`:
   - Check top reranked score after BGE reranking
   - If score < 0.4, trigger second pass:
     - Strip stopwords from query
     - Extract 2-3 highest-signal nouns/entities
     - Re-run hybrid search
   - If second pass also < 0.4, skip LLM
   - Return "no relevant documents" with reformulation suggestions
2. Add `_extract_signal_terms()` method
3. Add `_generate_reformulation_suggestions()` method

**Files to Modify:**
- `src/retrieval/pipeline.py`

---

### Fix 7: Reranking Diversity
**Priority:** MEDIUM

**Implementation Plan:**
1. Update `retrieval/pipeline.py` or `bge_reranker.py`:
   - After BGE reranking produces top 15
   - Apply source cap: max 3 chunks from any single `doc_id`
   - Skip 4th+ chunk from same doc, take next from different doc
2. Add `_apply_source_diversity()` method

**Files to Modify:**
- `src/retrieval/bge_reranker.py` (preferred)
- OR `src/retrieval/pipeline.py`

---

### Fix 8: Schema Normalization Trap
**Priority:** MEDIUM

**Implementation Plan:**
1. Update `prompt_templates.py`:
   - Add concrete JSON example to every prompt
   - Show exact field names with real-looking content
   - Not schema definition, literal example
2. Test with 20 queries
3. Remove normalization code from `answer_generator.py` once reliable

**Files to Modify:**
- `src/generation/prompt_templates.py`
- `src/generation/answer_generator.py` (remove normalization after testing)

---

### Fix 9: Confidence Field Behavior
**Priority:** MEDIUM

**Implementation Plan:**
1. Update `answer_generator.py`:
   - Calculate actual confidence based on:
     - Top chunk score
     - Number of chunks used
     - Answer length
   - Three tiers:
     - High: respond normally
     - Medium: append verification note
     - Low/None: skip LLM, return "no reliable information" message
2. Add `_calculate_confidence()` method
3. Add `_generate_reformulation_suggestions()` method

**Files to Modify:**
- `src/generation/answer_generator.py`

---

### Fix 10: Context Window Tracking
**Priority:** LOW

**Implementation Plan:**
1. Update `answer_generator.py`:
   - Add `tokens_used` and `chunks_used` to response metadata
   - Log to `logs/context_usage.jsonl` (append-only)
2. Create log analysis script:
   - Read `context_usage.jsonl`
   - Calculate percentiles (p50, p90, p99)
   - Identify if limits are too loose/tight
3. Add logging directory to `.gitignore`

**Files to Modify:**
- `src/generation/answer_generator.py`
- Create `scripts/analyze_context_usage.py`
- Update `.gitignore`

---

### Fix 11: Rate Limiting
**Priority:** MEDIUM

**Implementation Plan:**
1. Update `api/main.py`:
   - Add `asyncio.Semaphore(3)` for LLM concurrency
   - Add per-IP rate limiter (20 req/min) using sliding window
   - Return 429 Too Many Requests with Retry-After header
2. Create `src/utils/rate_limiter.py`:
   - `RateLimiter` class with sliding window counter
   - In-memory storage (no Redis needed)
   - Automatic cleanup of old entries

**Files to Modify:**
- `src/api/main.py`
- Create `src/utils/rate_limiter.py`

---

### Fix 12: Demo Auth
**Priority:** HIGH (Security)

**Implementation Plan:**
1. Move credentials to `.env`:
   ```
   DEMO_USER_STUDENT=student@nmims.edu:hashed_password
   DEMO_USER_FACULTY=faculty@nmims.edu:hashed_password
   DEMO_USER_ADMIN=admin@nmims.edu:hashed_password
   ```
2. Update `api/main.py`:
   - Load credentials from env
   - Use bcrypt for password hashing
   - Keep JWT token generation
3. Add `bcrypt==4.1.2` to requirements.txt
4. Ensure `.env` in `.gitignore`

**Files to Modify:**
- `Faculty Part/.env`
- `src/api/main.py`
- `requirements.txt`
- `.gitignore`

---

## 🔄 Integration Tasks

After individual fixes are complete:

1. **Update all imports** - Remove semantic_chunker references
2. **Rebuild BM25 index** - With new tokenizer
3. **Run evaluation baseline** - Before and after each fix
4. **Update ARCHITECTURE.md** - Document new approaches
5. **Test end-to-end** - Full ingestion → retrieval → generation flow

---

## 📊 Testing Checklist

Before marking any fix as complete:

- [ ] Unit tests pass (if applicable)
- [ ] Evaluation baseline shows improvement or no regression
- [ ] Manual testing with 5+ queries
- [ ] No new errors in logs
- [ ] Performance impact measured (if applicable)

---

## 🎯 Priority Order

**Week 1:**
1. Fix 1 (Chunking) - Complete implementation
2. Fix 2 (Token limits) - Sentence boundaries
3. Fix 12 (Auth) - Security critical
4. Fix 4 (Title stripping) - Complete integration
5. Fix 6 (BM25) - Complete integration

**Week 2:**
6. Fix 3 (Citations) - UX improvement
7. Fix 5 (Single-shot) - Accuracy improvement
8. Fix 7 (Diversity) - Quality improvement
9. Fix 11 (Rate limiting) - System protection

**Week 3:**
10. Fix 8 (Schema) - Reliability improvement
11. Fix 9 (Confidence) - UX improvement
12. Fix 10 (Tracking) - Observability

---

**Last Updated:** 2026-03-26
