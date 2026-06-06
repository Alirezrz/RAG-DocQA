import re
import math
import json
from collections import Counter

from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer

#  Singleton embedding model:

_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading embedding model... (first time only)")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded!")
    return _model


#  STEP 1 — Extract rich structure from .docx

def extract_structured_paragraphs(file_path):
    """
    Returns a list of dicts:
        [{'type': 'heading'|'text', 'text': '...'}, ...]
    """
    doc = DocxDocument(file_path)
    paragraphs = []

    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue  

        style_name = p.style.name.lower() if p.style else ''
        is_heading  = 'heading' in style_name

        paragraphs.append({
            'type': 'heading' if is_heading else 'text',
            'text': text,
        })

    return paragraphs


#  STEP 2 — Structure-Aware + Semantic Chunking

SEMANTIC_SPLIT_THRESHOLD = 0.45
MAX_CHUNK_WORDS = 600


def semantic_chunk(paragraphs: list):
    if not paragraphs:
        return []

    model = get_model()
    texts = [p['text'] for p in paragraphs]
    embeddings = model.encode(texts, show_progress_bar=False)  # shape: (N, 384)

    chunks= []
    current_group = []         
    current_words = 0

    def flush_group():
        """Turn current_group into a chunk string and reset."""
        if current_group:
            chunk_text = '\n'.join(p['text'] for p in current_group)
            chunks.append(chunk_text)
        current_group.clear()

    for i, para in enumerate(paragraphs):
        word_count = len(para['text'].split())

        if not current_group:
            current_group.append(para)
            current_words = word_count
            continue


        split = False

        # Signal 1 — STRUCTURAL: a heading always starts a new chunk
        if para['type'] == 'heading':
            split = True

        # Signal 2 — SEMANTIC: meaning shifted between this para and the last
        if not split:
            prev_embedding = embeddings[i - 1]
            curr_embedding = embeddings[i]

            # cosine similarity via numpy dot-product on unit vectors
            sim = float(
                (prev_embedding @ curr_embedding) /((prev_embedding @ prev_embedding) ** 0.5 *(curr_embedding @ curr_embedding) ** 0.5 + 1e-9))
            if sim < SEMANTIC_SPLIT_THRESHOLD:
                split = True
                print(f"  [Chunker] Semantic split at para {i} (sim={sim:.2f})")

        # Signal 3 — SIZE: chunk is already big enough, force a cut
        if current_words + word_count > MAX_CHUNK_WORDS:
            split = True

        if split:
            chunk_text = '\n'.join(p['text'] for p in current_group)
            chunks.append(chunk_text)
            current_group = [para]
            current_words = word_count
        else:
            current_group.append(para)
            current_words += word_count

    if current_group:
        chunks.append('\n'.join(p['text'] for p in current_group))

    return chunks


# ─────────────────────────────────────────────
#  STEP 3 — BM25 term statistics (for hybrid search)
# ─────────────────────────────────────────────

def tokenize(text: str):
    return re.findall(r'[a-zA-Z0-9]+', text.lower())


def compute_bm25_stats(text: str, all_chunk_texts: list):

    tokens = tokenize(text)
    tf     = dict(Counter(tokens))        # {'word': frequency, ...}
    return tf


#  STEP 4 — Embedding

def generate_embedding(text: str) -> list:
    """Convert text to a 384-dimensional vector."""
    model  = get_model()
    vector = model.encode(text)
    return vector.tolist()


#  STEP 5 — Main entry point

def process_document(document):

    from documents.models import DocumentChunk

    print(f"\nProcessing: {document.title}")

    # 1. Extract structured paragraphs (with heading tags)
    paragraphs = extract_structured_paragraphs(document.file.path)
    full_text  = '\n'.join(p['text'] for p in paragraphs)

    document.extracted_text = full_text
    document.save()
    print(f"  Extracted {len(paragraphs)} paragraphs ({len(full_text)} chars)")

    document.chunks.all().delete()

    chunk_texts = semantic_chunk(paragraphs)
    print(f"  Created {len(chunk_texts)} smart chunks")

    for index, chunk_text in enumerate(chunk_texts):
        embedding = generate_embedding(chunk_text)
        bm25_tf   = compute_bm25_stats(chunk_text, chunk_texts)  # term frequencies

        chunk = DocumentChunk(
            document    = document,
            content     = chunk_text,
            chunk_index = index,
        )
        chunk.set_embedding(embedding)
        chunk.set_bm25_tf(bm25_tf)     
        chunk.save()

    print(f"  Done! Saved {len(chunk_texts)} chunks with embeddings + BM25 stats")