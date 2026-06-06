import re
import math
import time
import numpy as np
from collections import Counter
from django.conf import settings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from documents.processing import generate_embedding, tokenize


#  Dense (vector) helpers

def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """
    1.0 = identical meaning, 0.0 = unrelated, -1.0 = opposite.
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


#  BM25 helpers

K1 = 1.5 # term-frequency saturation: higher -> more weight to tf
B  = 0.75  # length normalisation: 1.0 = full normalisation, 0 = none


def compute_bm25_scores(query_tokens: list, chunks: list) -> list:

    N = len(chunks)
    if N == 0:
        return []

    tf_maps = []
    lengths = []
    for chunk in chunks:
        tf = chunk.get_bm25_tf()
        tf_maps.append(tf)
        lengths.append(sum(tf.values()) or 1)  

    avgdl = sum(lengths) / N

    df = Counter()
    for t in set(query_tokens):
        for tf in tf_maps:
            if t in tf:
                df[t] += 1

    scores = []
    for i, tf in enumerate(tf_maps):
        score = 0.0
        dl = lengths[i]

        for t in query_tokens:
            if t not in tf:
                continue                   

            tf_val = tf[t]
            df_val  = df.get(t, 0)

            idf = math.log((N - df_val + 0.5) / (df_val + 0.5) + 1)
            tf_norm = (tf_val * (K1 + 1)) / (
                tf_val + K1 * (1 - B + B * dl / avgdl)
            )

            score += idf * tf_norm

        scores.append(score)

    return scores


#  Reciprocal Rank Fusion

RRF_K = 60   # standard constant; higher -> diminishes rank advantage


def reciprocal_rank_fusion(dense_ranks: list, bm25_ranks: list) -> list:
    rrf_scores = {}

    for rank, idx in enumerate(dense_ranks):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

    for rank, idx in enumerate(bm25_ranks):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

    # Sort chunk indices by combined RRF score, highest first
    return sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)


# main function
def find_relevant_chunks(question: str, top_k: int = 4):
    from documents.models import DocumentChunk

    chunks =list(
        DocumentChunk.objects.filter(
            embedding_json__isnull=False
        ).select_related('document')
    )

    if not chunks:
        return []

    #DENSE: embed question, score every chunk
    question_vector = generate_embedding(question)
    dense_scores = []
    for chunk in chunks:
        vec= chunk.get_embedding()
        score= cosine_similarity(question_vector, vec) if vec else 0.0
        dense_scores.append(score)

    dense_ranks = sorted(range(len(chunks)),key=lambda i: dense_scores[i], reverse=True)

    #BM25: tokenize question, score every chunk
    query_tokens = tokenize(question)
    bm25_scores= compute_bm25_scores(query_tokens, chunks)
    bm25_ranks= sorted(range(len(chunks)),key=lambda i: bm25_scores[i],  reverse=True)

    print(f"  [Hybrid] Dense top-1: chunk {dense_ranks[0]} "f"(score={dense_scores[dense_ranks[0]]:.3f})")
    print(f"  [Hybrid] BM25  top-1: chunk {bm25_ranks[0]}  "f"(score={bm25_scores[bm25_ranks[0]]:.3f})")

    #RRF
    fused_ranks = reciprocal_rank_fusion(dense_ranks, bm25_ranks)

    return [chunks[i] for i in fused_ranks[:top_k]]


#  LLM client

def get_llm():
    return ChatOpenAI(
        openai_api_key   = settings.OPENROUTER_API_KEY,
        openai_api_base  = "https://openrouter.ai/api/v1",
        model_name       = settings.OPENROUTER_MODEL,
        temperature      = 0.1,
        max_tokens       = 1000,
        default_headers  = {
            "HTTP-Referer": "http://localhost:8000",
            "X-Title":      "DocQA",
        }
    )



def ask_question(question: str):
    """
    End-to-end RAG:  question -> hybrid retrieval -> LLM -> answer.
    """
    start_time = time.time()
    print(f"\n[Pipeline] Question: {question}")

    chunks = find_relevant_chunks(question, top_k=4)
    print(f"[Pipeline] Retrieved {len(chunks)} chunks via hybrid search")

    if chunks:
        context_parts = []
        for chunk in chunks:
            context_parts.append(
                f"[From: {chunk.document.title}, section {chunk.chunk_index}]\n"
                f"{chunk.content}"
            )
        context = "\n\n---\n\n".join(context_parts)
    else:
        context = "No relevant documents found."

    messages = [
        SystemMessage(content=(
            "You are a helpful assistant that answers questions "
            "based strictly on the provided document context. "
            "If the answer is not in the context, say so honestly. "
            "Do not make up information. Be concise and direct."
        )),
        HumanMessage(content=(
            f"Context from documents:\n\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer based only on the context above:"
        ))
    ]

    llm= get_llm()
    print(f"[Pipeline] Calling {settings.OPENROUTER_MODEL}...")
    response = llm.invoke(messages)
    answer= response.content
    print(f"[Pipeline] Got answer ({len(answer)} chars)")

    source_docs = list({chunk.document for chunk in chunks})

    return {
        'answer':                 answer,
        'source_documents':       source_docs,
        'chunks_used':            len(chunks),
        'response_time_seconds':  round(time.time() - start_time, 2),
        'model_used':             settings.OPENROUTER_MODEL,
    }