# Semantic Chunking Strategy for Faculty Resources

## Core Principle

**Chunk by meaning and completeness, not by token count.**

A chunk is only valid if it can answer a question on its own without needing surrounding context to make sense.

## What Makes Faculty Content Unique

### 1. PROCEDURES are sequential
Steps must stay together. If a procedure has 5 steps, all 5 live in one chunk. Never split a process mid-way.

### 2. RULES are conditional
Keep the condition and consequence together. "If X then Y" must never be split into separate chunks.

### 3. FORMS have field-level instructions
Keep all instructions for a single form together as one chunk.

### 4. CIRCULARS have dates
Newer documents on the same topic override older ones. Flag this relationship at ingestion time.

### 5. POLICIES have scope
"Who this applies to" must always stay attached to the policy content, never separated.

## Three Chunk Levels

### LEVEL 1 — DOCUMENT OVERVIEW
One chunk per document.

**Captures:** what this document is about, who it applies to, and what topics it covers.

**Use when:** faculty asks a broad or exploratory question.

### LEVEL 2 — COMPLETE PROCEDURE OR POLICY SECTION
One chunk per complete process, rule set, or section. This is the primary retrieval unit.

**Use when:** faculty asks how to do something or what the rules are.

**Size constraint:** keep under 600 tokens, but never break meaning to hit that limit.

### LEVEL 3 — SINGLE RULE, DEADLINE, OR DEFINITION
One chunk per standalone fact.

**Examples:**
- "Casual leave entitlement is 12 days per year"
- "Submit application 7 days before leave starts"

**Use when:** faculty asks a direct factual question.

## Splitting Rules

### SPLIT at:
- Document headings and section boundaries
- Where one complete topic ends and another begins
- Before a new procedure starts

### NEVER SPLIT:
- Mid-procedure (all steps stay together)
- A rule from its condition or exception
- A form's instructions across multiple chunks
- A definition from its explanation
- A deadline from the action it applies to

## Overlap

Apply a small overlap (around 50 tokens) between Level 2 chunks so that context near a boundary isn't lost.

No overlap needed for Level 1 or Level 3.

## Special Content Types

### IMAGES AND DIAGRAMS
Extract text via OCR. Additionally write a plain-text description of what the image shows (flowchart, org chart, form layout etc.) and store that description as a chunk linked to its parent topic.

### PDFS WITH MIXED CONTENT
Process text regions and image regions separately, then associate image-derived chunks with the surrounding text topic.

### JSON FILES
Flatten into readable key-value descriptions before chunking. Do not store raw JSON as a chunk — it retrieves poorly.

### OUTDATED DOCUMENTS
When a newer document covers the same topic as an older one, mark the older chunks as superseded. Exclude superseded chunks from retrieval unless explicitly requested.

## Intent-Based Retrieval Routing

Before running search, classify the faculty query into one of these intent types and adjust retrieval accordingly:

### LOOKUP INTENT
"what is the X limit", "what does X mean"
→ Target Level 3 atomic chunks first

### PROCEDURE INTENT
"how do I apply for X", "what are the steps for X"
→ Target Level 2 procedure chunks

### ELIGIBILITY INTENT
"can I apply for X", "am I allowed to X"
→ Target Level 2 rule chunks, filter by applicability scope

### FORM/APPLICATION INTENT
"help me fill X form", "what do I write in X"
→ Target form instruction chunks + generate a draft using retrieved content

### GENERAL INTENT
Broad or unclear query
→ Start with Level 1 summary, then expand to Level 2 if needed

## Retrieval Pipeline Order

1. Detect query intent
2. Apply metadata filter for relevant domain/document type
3. Run hybrid search (vector similarity + keyword BM25 together)
4. Take top 20 results
5. Re-rank using cross-encoder to get top 5
6. Pass top 5 chunks + original query to LLM
7. LLM generates answer grounded only in retrieved chunks

## LLM Answer Behavior

The LLM should:
- Answer based strictly on retrieved chunks, not assumptions
- If a procedure has steps, present them in order
- If a rule has conditions, state the condition clearly first
- If the answer is not found in chunks, say so explicitly rather than guessing
- Keep answers concise but complete — faculty need actionable answers
