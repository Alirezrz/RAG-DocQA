# Doc-QA — Document Question Answering System

A production-grade **Retrieval-Augmented Generation (RAG)** API built with Django. Upload Word documents, ask questions in plain English, get answers grounded strictly in your documents — with full source attribution.

---

## Table of Contents

1. [What This Is](#what-this-is)
2. [How It Actually Works — The Full Picture](#how-it-actually-works)
3. [The Math Behind Everything](#the-math-behind-everything)
4. [Project Structure](#project-structure)
5. [Setup & Installation](#setup--installation)
6. [Running the Server](#running-the-server)
7. [API Reference & Usage](#api-reference--usage)
8. [End-to-End Example](#end-to-end-example)
9. [Configuration](#configuration)
10. [What's Next](#whats-next)

---

## What This Is

Most LLMs like GPT-4 are trained on general internet data. They know a lot, but they don't know *your* documents — your contracts, research papers, internal reports, manuals. If you ask them about your content, they either hallucinate or admit they don't know.

**RAG (Retrieval-Augmented Generation)** solves this by giving the LLM exactly the right pieces of your document at query time, so it can answer questions grounded in real content rather than guessing.

This project implements a full RAG pipeline with two major upgrades over naive implementations:

- **Semantic + Structure-Aware Chunking** — understands document structure instead of blindly cutting every 500 words
- **Hybrid Search (Dense + BM25 + RRF)** — combines semantic understanding with keyword matching for dramatically better retrieval

---

## How It Actually Works

There are two completely separate phases. Understanding this separation is the key to understanding the whole system.

---

### Phase 1 — Indexing (happens once, when you upload a document)

When you POST a document to the API, the following pipeline runs automatically:

```
.docx file
    │
    ▼
[1] Text + Structure Extraction
    Read every paragraph, tag each one as either
    'heading' or 'text' using the .docx style metadata.
    │
    ▼
[2] Smart Chunking
    Walk through paragraphs one by one.
    At each boundary, ask three questions:
      • Is the next paragraph a heading?     → split (structural signal)
      • Did the meaning just shift?          → split (semantic signal)
      • Is the chunk already 600+ words?     → split (size safety net)
    Result: a list of coherent text chunks, each about one topic.
    │
    ▼
[3] Embedding (Dense Vectors)
    Pass each chunk through all-MiniLM-L6-v2.
    Output: a 384-dimensional vector per chunk.
    Similar meaning → similar vector → close in vector space.
    │
    ▼
[4] BM25 Indexing (Sparse / Keyword)
    Tokenize each chunk, count term frequencies.
    Store the term-frequency map per chunk.
    This is the keyword index used by BM25 at query time.
    │
    ▼
[5] Persist to Database
    Save chunk text + embedding vector + BM25 term frequencies
    to SQLite. The document is now fully indexed.
```

This happens **once per document**, triggered automatically by Django's `post_save` signal. You never call it manually.

---

### Phase 2 — Querying (happens every time you ask a question)

```
User question: "When does the contract expire?"
    │
    ▼
[1] Embed the question
    Pass it through the same MiniLM model.
    Output: a 384-dim vector representing the question's meaning.
    │
    ▼
[2] Dense Retrieval
    Compute cosine similarity between the question vector
    and every chunk vector stored in the database.
    Sort chunks by score → dense ranking list.
    │
    ▼
[3] BM25 Retrieval
    Tokenize the question: ["when", "does", "contract", "expire"]
    Score every chunk using the BM25 formula.
    Sort chunks by score → BM25 ranking list.
    │
    ▼
[4] Hybrid Fusion (RRF)
    Combine both ranking lists using Reciprocal Rank Fusion.
    Score = 1/(60 + rank_dense) + 1/(60 + rank_bm25)
    The chunks that appear near the top in BOTH lists win.
    Sort by fused score → final ranking.
    │
    ▼
[5] Build Context
    Take the top 4 chunks from the fused ranking.
    Format them as a context block:
    [From: Contract.docx, section 3]
    The agreement terminates on December 31, 2025...
    │
    ▼
[6] LLM Generation
    Send a structured prompt to the LLM via OpenRouter:
      SYSTEM: Answer only from the provided context.
              If the answer isn't there, say so.
      USER:   Context: {top 4 chunks}
              Question: {user's question}
    │
    ▼
[7] Return Answer
    Save the Q&A to history, return JSON with:
    - answer text
    - source document titles
    - timestamp
```

---

## The Math Behind Everything

### Cosine Similarity (Dense Retrieval)

Every chunk and every question gets encoded as a vector in 384-dimensional space. To find how semantically similar two texts are, we measure the angle between their vectors — not the distance. This is cosine similarity:

```
                  A · B
cos(θ) =  ─────────────────
            ‖A‖ × ‖B‖
```

Where:
- `A · B` is the dot product: Σ(Aᵢ × Bᵢ) for all 384 dimensions
- `‖A‖` is the magnitude (length) of vector A: √(Σ Aᵢ²)
- `‖B‖` is the magnitude of vector B

**Why angle and not distance?**
Distance penalizes long documents just for being long — they produce larger vectors. Cosine similarity normalizes by magnitude, so a short chunk and a long chunk can be equally relevant to a question as long as they talk about the same thing.

Result range:
```
 1.0  →  identical meaning
 0.0  →  completely unrelated
-1.0  →  opposite meaning (rare in practice)
```

In code:
```python
dot_product = np.dot(question_vec, chunk_vec)
similarity  = dot_product / (np.linalg.norm(question_vec) * np.linalg.norm(chunk_vec))
```

---

### BM25 (Keyword Retrieval)

BM25 (Best Match 25) is the classical information retrieval formula. It scores how relevant a chunk `d` is to a query by summing contributions from each query term `t`:

```
              N - df(t) + 0.5              tf(t,d) × (k₁ + 1)
score(d,q) = Σ  log( ─────────────────── + 1 ) × ──────────────────────────────────
              t      df(t) + 0.5               tf(t,d) + k₁ × (1 - b + b × |d|/avgdl)
```

Breaking down every symbol:

| Symbol | Meaning |
|--------|---------|
| `N` | Total number of chunks in the database |
| `df(t)` | How many chunks contain term `t` (document frequency) |
| `tf(t,d)` | How many times term `t` appears in chunk `d` (term frequency) |
| `\|d\|` | Length of chunk `d` in tokens |
| `avgdl` | Average chunk length across all chunks |
| `k₁ = 1.5` | Term frequency saturation — higher means more weight to frequency |
| `b = 0.75` | Length normalization — 1.0 = full normalization, 0 = none |

**The IDF part** (left of the ×):
```
      N - df(t) + 0.5
log( ─────────────── + 1 )
      df(t) + 0.5
```
This is **Inverse Document Frequency**. Words that appear in every chunk ("the", "is", "a") get IDF ≈ 0 — they tell us nothing. Words that appear in only 2 out of 500 chunks ("arbitration", "photosynthesis") get high IDF — they're discriminating.

**The TF_norm part** (right of the ×):
```
tf(t,d) × (k₁ + 1)
────────────────────────────────────────
tf(t,d) + k₁ × (1 - b + b × |d|/avgdl)
```
This normalizes term frequency by chunk length. A chunk that says "contract" 5 times in 100 words is more relevant than one that says "contract" 5 times in 2000 words. The `b` parameter controls how aggressively we penalize long chunks.

**Why use BM25 at all if we have embeddings?**

Embeddings compress meaning into 384 numbers. In that compression, rare technical terms — article numbers, proper nouns, codes — can get washed out. BM25 never loses a keyword. It scores zero for a missing term and high for an exact match. The two methods cover each other's blind spots.

---

### Semantic Chunking (Cosine at Paragraph Level)

During indexing, we embed every paragraph with MiniLM and compare adjacent paragraphs:

```
similarity(paraᵢ, paraᵢ₊₁) = cosine(embed(paraᵢ), embed(paraᵢ₊₁))
```

If `similarity < 0.45` → the meaning shifted → split here.

This threshold (0.45) was chosen empirically. Lower it to get fewer, larger chunks. Raise it to get more, smaller chunks. The intuition: paragraphs continuing the same thought typically score 0.6–0.9. Paragraphs starting a new topic drop below 0.5.

```python
SEMANTIC_SPLIT_THRESHOLD = 0.45  # in processing.py — tune this freely
MAX_CHUNK_WORDS = 600            # hard cap regardless of similarity
```

---

### Reciprocal Rank Fusion (Hybrid Search Fusion)

After dense retrieval gives us one ranked list and BM25 gives us another, we need to combine them. We can't just add the raw scores — cosine similarity lives in [-1, 1] while BM25 scores are unbounded floats (could be 0.2 or 45.0). Adding them raw lets BM25 dominate just because its numbers are bigger.

**RRF works on ranks, not scores.** Ranks are scale-free: rank 1 means the same thing regardless of which system produced it.

```
           1                    1
RRF(d) = ─────────────── + ───────────────
          K + rank_dense    K + rank_bm25

K = 60  (standard constant)
```

The K=60 constant softens the advantage of being rank 1 vs rank 2. Without it, rank 1 gets score 1.0 and rank 2 gets 0.5 — a huge cliff. With K=60, rank 1 gets 1/61 ≈ 0.0164 and rank 2 gets 1/62 ≈ 0.0161 — much more gradual.

**Example with 3 chunks (A, B, C):**

```
Dense ranking:   A=1,  B=2,  C=3
BM25  ranking:   B=1,  A=2,  C=3

RRF(A) = 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
RRF(B) = 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252
RRF(C) = 1/(60+3) + 1/(60+3) = 0.01587 + 0.01587 = 0.03175
```

A and B are tied — each appears near the top of one list. C is last in both — lowest combined score. This is exactly right: the system rewards chunks that multiple retrieval methods agree on.

---

## Project Structure

```
Doc-QA/
│
├── core/                        # Django project config
│   ├── settings.py              # DB, installed apps, OpenRouter keys
│   ├── urls.py                  # Root URL routing
│   └── wsgi.py
│
├── documents/                   # Ingestion app
│   ├── models.py                # Document + DocumentChunk (with embeddings + BM25)
│   ├── processing.py            # The full indexing pipeline
│   │   ├── extract_structured_paragraphs()   # .docx → tagged paragraphs
│   │   ├── semantic_chunk()                  # Smart chunking
│   │   ├── compute_bm25_stats()              # Term frequency indexing
│   │   ├── generate_embedding()              # MiniLM encoding
│   │   └── process_document()               # Orchestrates all of the above
│   ├── serializers.py           # REST serializer (includes chunk_count)
│   ├── signals.py               # Triggers process_document() on upload
│   ├── views.py                 # DocumentViewSet (full CRUD)
│   └── urls.py                  # /api/documents/
│
├── qa/                          # Query app
│   ├── models.py                # QAHistory (stores every Q&A + source docs)
│   ├── pipeline.py              # The full query pipeline
│   │   ├── cosine_similarity()              # Dense scoring
│   │   ├── compute_bm25_scores()            # BM25 scoring
│   │   ├── reciprocal_rank_fusion()         # RRF fusion
│   │   ├── find_relevant_chunks()           # Hybrid retrieval (all three above)
│   │   └── ask_question()                   # Full RAG: retrieve → prompt → LLM
│   ├── serializers.py           # QAHistorySerializer
│   ├── views.py                 # AskView (POST) + HistoryView (GET)
│   └── urls.py                  # /api/ask/ and /api/history/
│
├── .env.example                 # Copy this to .env and fill in your keys
├── requirements.txt
└── manage.py
```

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- pip

### Step 1 — Clone and enter the project

```bash
git clone <your-repo-url>
cd Doc-QA
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---------|---------|
| `django` | Web framework, ORM, signals |
| `djangorestframework` | REST API layer (ViewSets, serializers) |
| `python-docx` | Reads .docx files, exposes paragraphs and styles |
| `sentence-transformers` | Loads MiniLM, encodes text to 384-dim vectors |
| `langchain` + `langchain-openai` | LLM client abstraction (OpenRouter integration) |
| `numpy` | Vector math (dot products, norms) |
| `python-dotenv` | Loads .env file into environment variables |

The first time `sentence-transformers` runs, it downloads the `all-MiniLM-L6-v2` model (~90MB) from HuggingFace. This is automatic and only happens once.

### Step 4 — Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
SECRET_KEY=replace-this-with-a-long-random-string
DEBUG=True
OPENROUTER_API_KEY=sk-or-v1-your-key-from-openrouter.ai
OPENROUTER_MODEL=mistralai/mistral-7b-instruct
```

Get your OpenRouter API key at [openrouter.ai](https://openrouter.ai). The model string must be a valid OpenRouter model identifier. Some good free/cheap options:

```
mistralai/mistral-7b-instruct
meta-llama/llama-3-8b-instruct
google/gemma-7b-it
```

### Step 5 — Run database migrations

```bash
python manage.py migrate
```

This creates all tables including `documents_document`, `documents_documentchunk`, and `qa_qahistory`.

---

## Running the Server

```bash
python manage.py runserver
```

The server starts at `http://localhost:8000`.

You can browse the API directly in your browser at `http://localhost:8000/api/` — Django REST Framework provides a web UI out of the box.

---

## API Reference & Usage

### Upload a Document

```
POST /api/documents/
Content-Type: multipart/form-data
```

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `title` | string | yes | A human-readable name for the document |
| `file` | file | yes | A `.docx` Word document |

**Example with curl:**

```bash
curl -X POST http://localhost:8000/api/documents/ \
  -F "title=Employment Contract" \
  -F "file=@/path/to/contract.docx"
```

**Example with Postman:**
- Method: POST
- URL: `http://localhost:8000/api/documents/`
- Body: form-data
  - `title` → `Employment Contract`
  - `file` → select your .docx file

**Response (201 Created):**

```json
{
    "id": 1,
    "title": "Employment Contract",
    "file": "/media/documents/contract.docx",
    "extracted_text": "This Employment Agreement is entered into...",
    "uploaded_at": "2026-06-06T19:00:00Z",
    "updated_at": "2026-06-06T19:00:00Z",
    "chunk_count": 12
}
```

The `chunk_count` tells you how many chunks the document was split into. The indexing pipeline (extract → chunk → embed → BM25) ran automatically in the background.

> **Note:** The first upload takes longer because the MiniLM model needs to load into memory. Subsequent uploads are faster.

---

### List All Documents

```
GET /api/documents/
```

```bash
curl http://localhost:8000/api/documents/
```

Returns a list of all uploaded documents ordered by upload date (newest first).

---

### Get a Single Document

```
GET /api/documents/{id}/
```

```bash
curl http://localhost:8000/api/documents/1/
```

---

### Delete a Document

```
DELETE /api/documents/{id}/
```

```bash
curl -X DELETE http://localhost:8000/api/documents/1/
```

Returns `204 No Content`. The document, its file, and all its chunks are deleted.

---

### Ask a Question

```
POST /api/ask/
Content-Type: application/json
```

**Body:**

```json
{
    "question": "What are the payment terms in the contract?"
}
```

**Example with curl:**

```bash
curl -X POST http://localhost:8000/api/ask/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the payment terms in the contract?"}'
```

**Response (201 Created):**

```json
{
    "id": 1,
    "question": "What are the payment terms in the contract?",
    "answer": "According to the contract, payment is due within 30 days of invoice. Late payments incur a 2% monthly interest charge.",
    "sources": [
        "Employment Contract"
    ],
    "created_at": "2026-06-06T19:30:00Z"
}
```

| Field | What it means |
|-------|---------------|
| `answer` | The LLM's response, grounded in your documents |
| `sources` | Which document(s) the relevant chunks came from |
| `created_at` | When this Q&A was recorded |

If the answer is not found in any document, the LLM will say so explicitly — it is instructed not to make things up.

---

### View Question History

```
GET /api/history/
```

```bash
curl http://localhost:8000/api/history/
```

Returns all past questions and answers, newest first:

```json
[
    {
        "id": 3,
        "question": "When does the contract expire?",
        "answer": "The contract expires on December 31, 2025.",
        "sources": ["Employment Contract"],
        "created_at": "2026-06-06T19:35:00Z"
    },
    {
        "id": 2,
        "question": "What is the notice period?",
        "answer": "The notice period is 30 days for either party.",
        "sources": ["Employment Contract"],
        "created_at": "2026-06-06T19:32:00Z"
    }
]
```

---

## End-to-End Example

Here is the complete flow from zero to answered question:

**1. Start the server:**
```bash
python manage.py runserver
```

**2. Upload a document:**
```bash
curl -X POST http://localhost:8000/api/documents/ \
  -F "title=World History 20th Century" \
  -F "file=@history.docx"
```

You'll see in the server console:
```
Processing: World History 20th Century
  Extracted 47 paragraphs (18432 chars)
Loading embedding model... (first time only)
Model loaded!
  [Chunker] Semantic split at para 8 (sim=0.31)
  [Chunker] Semantic split at para 19 (sim=0.28)
  [Chunker] Semantic split at para 31 (sim=0.38)
  Created 9 smart chunks
  Done! Saved 9 chunks with embeddings + BM25 stats
```

**3. Ask a question:**
```bash
curl -X POST http://localhost:8000/api/ask/ \
  -H "Content-Type: application/json" \
  -d '{"question": "When did World War 1 start?"}'
```

Server console:
```
[Pipeline] Question: When did World War 1 start?
  [Hybrid] Dense top-1: chunk 2 (score=0.847)
  [Hybrid] BM25  top-1: chunk 2 (score=12.431)
[Pipeline] Retrieved 4 chunks via hybrid search
[Pipeline] Calling mistralai/mistral-7b-instruct...
[Pipeline] Got answer (52 chars)
```

Response:
```json
{
    "id": 1,
    "question": "When did World War 1 start?",
    "answer": "World War I started on July 28, 1914.",
    "sources": ["World History 20th Century"],
    "created_at": "2026-06-06T19:30:34Z"
}
```

**4. Check history:**
```bash
curl http://localhost:8000/api/history/
```

---

## Configuration

All configuration lives in `.env`. No code changes needed to switch models or tune behavior.

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `SECRET_KEY` | Django secret key | any long random string |
| `DEBUG` | Debug mode | `True` or `False` |
| `OPENROUTER_API_KEY` | Your OpenRouter API key | `sk-or-v1-...` |
| `OPENROUTER_MODEL` | Model to use for generation | `mistralai/mistral-7b-instruct` |

### Tunable Parameters in Code

These constants control retrieval quality. Edit them directly in the source files:

**`documents/processing.py`**

```python
SEMANTIC_SPLIT_THRESHOLD = 0.45  # Lower → fewer, bigger chunks
                                  # Higher → more, smaller chunks

MAX_CHUNK_WORDS = 600             # Hard cap per chunk
```

**`qa/pipeline.py`**

```python
K1 = 1.5   # BM25 term frequency weight. Higher → more emphasis on repetition
B  = 0.75  # BM25 length normalization. 0 = ignore length, 1 = full normalization
RRF_K = 60 # Fusion constant. Higher → gentler rank advantage for top results
```

**`qa/views.py` → `find_relevant_chunks(question, top_k=4)`**

Change `top_k` to pass more or fewer chunks to the LLM. More chunks = more context but higher token cost and risk of confusion.

---

## What's Next

The system is production-capable for small to medium document sets. Here are the natural next steps in order of impact:

**Cross-encoder re-ranking** — After retrieving the top 10 chunks with hybrid search, run a cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L6-v2`) to re-score and re-rank them before sending top 4 to the LLM. Cross-encoders compare the question and chunk together rather than separately, giving much more precise relevance judgement.

**Vector database** — Replace SQLite JSON storage with ChromaDB or FAISS. Currently all chunk embeddings are loaded into Python memory on every query. With thousands of chunks this becomes slow. A proper vector DB uses Approximate Nearest Neighbor (ANN) search to retrieve top candidates in milliseconds.

**Async processing** — Embedding generation runs synchronously inside the HTTP request. For large documents this can take 10–30 seconds and may time out. Moving it to Celery + Redis would let the upload endpoint return immediately and process in the background.

**Multi-format support** — Extend `extract_structured_paragraphs()` to handle PDF, TXT, and Markdown. The rest of the pipeline needs no changes.

**Stronger embedding model** — Swap `all-MiniLM-L6-v2` for `BAAI/bge-large-en-v1.5` in `get_model()`. Drop-in replacement, significantly better retrieval quality, especially on technical or domain-specific documents.

**Multi-turn conversation** — Currently each question is independent. Passing the last few Q&A pairs as conversation history to the LLM would allow follow-up questions like "tell me more about that".

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web framework | Django 4.2 + Django REST Framework |
| Embedding model | `all-MiniLM-L6-v2` via sentence-transformers |
| Keyword search | BM25 (implemented from scratch with NumPy) |
| Vector fusion | Reciprocal Rank Fusion (implemented from scratch) |
| LLM provider | OpenRouter (OpenAI-compatible API) |
| LLM client | LangChain `ChatOpenAI` adapter |
| Document parsing | python-docx |
| Database | SQLite (via Django ORM) |
| Vector storage | JSON fields in SQLite |

---

## License

MIT
