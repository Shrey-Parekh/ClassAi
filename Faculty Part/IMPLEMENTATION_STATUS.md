# Implementation Status: 13 Critical Fixes

## ✅ COMPLETED (7/13)

### Fix 1: Two Competing Chunking Strategies ✅
- Deleted `semantic_chunker.py`
- Updated `ingestion/pipeline.py` to use only `document_chunker.py`
- Added `chunk_level` assignment (overview/section/atomic)
- Single unified strategy

### Fix 2: Sentence Boundary Splitting ✅
- Updated `_split_by_size()` in `document_chunker.py`
- Walks backward to sentence boundaries (`. ` + capital)
- Never splits inside numbered lists
- Keeps lists whole even if oversized
- Added `_split_at_sentence_boundary()` method

### Fix 4: Title Stripping Inconsistency ✅
- Created `src/utils/query_normalizer.py`
- Single `QueryNormalizer.normalize_query()` function
- Centralized title stripping, whitespace, lowercase
- Ready for integration across pipeline

### Fix 6: BM25 Tokenization ✅
- Created `src/utils/bm25_tokenizer.py`
- NLTK word_tokenize with fallback
- Domain stopwords: nmims, university, college, document, section
- Standard English stopwords removed
- Added `nltk==3.8.1` to requirements

### Fix 11: Rate Limiting ✅
- Created `src/utils/rate_limiter.py`
- Sliding window algorithm (20 req/min per IP)
- In-memory storage, no Redis needed
- Integrated into `/query` endpoint
- Returns 429 with Retry-After header
- LLM semaphore (max 3 concurrent calls)

### Fix 12: Demo Auth ✅
- Moved credentials to `.env` file
- Bcrypt password hashing
- Added `bcrypt==4.1.2` to requirements
- Updated signin endpoint to use bcrypt.checkpw()
- Credentials format: `email:hash:role:name`

### Fix 13: Evaluation Baseline ✅
- Created `tests/eval_baseline.py`
- 7 test cases across categories
- Automated pass/fail scoring
- JSON output with timestamp
- Run: `python tests/eval_baseline.py`

---

## 🚧 INTEGRATION NEEDED (2/13)

### Fix 4: QueryNormalizer Integration
**Status:** Created but not integrated

**Remaining Work:**
1. Update `query_understanding.py`:
   - Replace `_strip_titles_for_embedding()` with `QueryNormalizer.normalize_query()`
   - Use in `_preprocess_query()`, `_normalize_query()`, `_expand_query()`

2. Update `retrieval/pipeline.py`:
   - Import `QueryNormalizer`
   - Use before embedding generation
   - Use before BM25 search

3. Update `hybrid_search.py`:
   - Use `QueryNormalizer` for query cleaning

**Files to Modify:**
- `src/retrieval/query_understanding.py`
- `src/retrieval/pipeline.py`
- `src/retrieval/hybrid_search.py`

### Fix 6: BM25Tokenizer Integration
**Status:** Created but not integrated

**Remaining Work:**
1. Update `hybrid_search.py`:
   - Import `BM25Tokenizer`
   - Replace `text.lower().split()` with `tokenizer.tokenize(text)`
   - Update `build_bm25_index()` to use tokenizer

2. Rebuild BM25 index after deployment

**Files to Modify:**
- `src/retrieval/hybrid_search.py`

---

## 📋 NOT STARTED (4/13)

### Fix 3: Citation Tracking
**Priority:** HIGH

**Implementation Plan:**
1. Update `answer_generator.py`:
   - Number chunks in context: `[1] chunk_text`, `[2] chunk_text`
   - Add to system prompt: "Annotate factual claims with [N]"
   - Parse output to extract citation numbers
   - Map numbers back to source metadata

2. Update `prompt_templates.py`:
   - Add citation instruction to all prompts
   - Show concrete example with inline citations

3. Update frontend `chat.js`:
   - Render `[N]` as superscripts
   - Link to source on click
   - Show source metadata in tooltip/modal

4. Update `chat.css`:
   - Style for citation superscripts
   - Hover effects

**Files to Create/Modify:**
- `src/generation/answer_generator.py`
- `src/generation/prompt_templates.py`
- `frontend/chat.js`
- `frontend/chat.css`

---

### Fix 5: Single-Shot Retrieval
**Priority:** MEDIUM

**Implementation Plan:**
1. Update `retrieval/pipeline.py`:
   - After reranking, check top score
   - If score < 0.4, trigger second pass:
     - Strip stopwords from query
     - Extract 2-3 signal nouns/entities
     - Re-run hybrid search
   - If second pass also < 0.4:
     - Skip LLM entirely
     - Return "no relevant documents" message
     - Suggest reformulation strategies

2. Add methods:
   - `_extract_signal_terms(query)` - Extract high-signal nouns
   - `_generate_reformulation_suggestions(query, intent)` - Suggest rephrasing

**Files to Modify:**
- `src/retrieval/pipeline.py`

---

### Fix 7: Reranking Diversity
**Priority:** MEDIUM

**Implementation Plan:**
1. Update `bge_reranker.py` or `pipeline.py`:
   - After BGE reranking produces top 15
   - Apply source cap: max 3 chunks from any single `doc_id`
   - Skip 4th+ chunk from same doc
   - Take next highest from different doc

2. Add method:
   - `_apply_source_diversity(results, max_per_doc=3)`

**Files to Modify:**
- `src/retrieval/bge_reranker.py` (preferred)
- OR `src/retrieval/pipeline.py`

---

### Fix 8: Schema Normalization Trap
**Priority:** MEDIUM

**Implementation Plan:**
1. Update `prompt_templates.py`:
   - Add concrete JSON example to every prompt
   - Show exact field names with real content
   - Not schema definition, literal example

2. Test with 20 queries

3. Remove normalization code from `answer_generator.py`:
   - Delete `_normalize_schema()` method
   - Delete `TYPE_ALIASES` and `SECTION_FIELD_FIXES`
   - Simplify `_parse_json_response()`

**Files to Modify:**
- `src/generation/prompt_templates.py`
- `src/generation/answer_generator.py` (after testing)

---

### Fix 9: Confidence Field Behavior
**Priority:** MEDIUM

**Implementation Plan:**
1. Update `answer_generator.py`:
   - Calculate actual confidence:
     - Based on top chunk score
     - Number of chunks used
     - Answer length
   - Three tiers:
     - High (>0.7): respond normally
     - Medium (0.4-0.7): append verification note
     - Low (<0.4): skip LLM, return "no reliable information"

2. Add methods:
   - `_calculate_confidence(chunks, scores)` - Calculate confidence score
   - `_generate_no_confidence_response(query, intent)` - Fallback message

**Files to Modify:**
- `src/generation/answer_generator.py`

---

### Fix 10: Context Window Tracking
**Priority:** LOW

**Implementation Plan:**
1. Update `answer_generator.py`:
   - Add `tokens_used` and `chunks_used` to response
   - Log to `logs/context_usage.jsonl` (append-only)
   - Format: `{"timestamp": "...", "query": "...", "tokens_used": 5000, "chunks_used": 10}`

2. Create `scripts/analyze_context_usage.py`:
   - Read `context_usage.jsonl`
   - Calculate percentiles (p50, p90, p99)
   - Identify if limits too loose/tight
   - Output recommendations

3. Update `.gitignore`:
   - Add `logs/`

**Files to Create/Modify:**
- `src/generation/answer_generator.py`
- Create `scripts/analyze_context_usage.py`
- `.gitignore`

---

## 🔄 NEXT STEPS

### Immediate (Week 1):
1. ✅ Complete Fix 1, 2, 11, 12, 13
2. Integrate QueryNormalizer (Fix 4)
3. Integrate BM25Tokenizer (Fix 6)
4. Rebuild BM25 index
5. Run evaluation baseline
6. Implement Fix 3 (Citations)

### Short-term (Week 2):
7. Implement Fix 5 (Single-shot retrieval)
8. Implement Fix 7 (Diversity)
9. Implement Fix 8 (Schema examples)
10. Run evaluation after each fix

### Medium-term (Week 3):
11. Implement Fix 9 (Confidence)
12. Implement Fix 10 (Tracking)
13. Final evaluation run
14. Update ARCHITECTURE.md

---

## 📊 Testing Checklist

Before deployment:
- [ ] Run evaluation baseline (before changes)
- [ ] Integrate QueryNormalizer
- [ ] Integrate BM25Tokenizer
- [ ] Rebuild BM25 index
- [ ] Run evaluation baseline (after changes)
- [ ] Manual testing with 10+ queries
- [ ] Check logs for errors
- [ ] Verify rate limiting works
- [ ] Verify bcrypt auth works
- [ ] Test with invalid credentials

---

## ⚠️ Known Issues

1. **QueryNormalizer not integrated** - Created but not used yet
2. **BM25Tokenizer not integrated** - Created but not used yet
3. **BM25 index needs rebuild** - After tokenizer integration
4. **No citation tracking** - Answers don't link to sources
5. **No second-pass retrieval** - Low-confidence queries go to LLM anyway
6. **No diversity cap** - Same document can dominate results
7. **Schema normalization still active** - Should be removed after prompt improvements

---

**Last Updated:** 2026-03-26
**Completion:** 7/13 fixes (54%)
