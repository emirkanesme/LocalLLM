import math
import re

from config import DB_NAME, MIN_SIMILARITY, TOP_K
from database import fetch_all_documents, get_metadata

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)
CAMEL_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])")


def tokenize(text):
    """Tokenize text; splits CamelCase so 'EsyslauncherPremium' → esys, launcher, premium."""
    normalized = CAMEL_SPLIT.sub(" ", text)
    return TOKEN_PATTERN.findall(normalized.lower())


def lexical_overlap_score(query_terms, content, doc_terms):
    """Substring / prefix overlap for compound words and short queries."""
    if not query_terms:
        return 0.0

    content_lower = content.lower()
    hits = 0.0
    for term in query_terms:
        if term in content_lower:
            hits += 1.0
            continue
        for doc_term in doc_terms:
            if term in doc_term or doc_term in term:
                hits += 0.7
                break

    return hits / len(query_terms)


def combined_similarity(tfidf_score, lexical_score):
    """Blend vector and lexical signals."""
    return 0.65 * tfidf_score + 0.35 * lexical_score


def term_frequencies(tokens):
    """Return {term: count} for a token list."""
    freqs = {}
    for token in tokens:
        freqs[token] = freqs.get(token, 0) + 1
    return freqs


def chunk_text(text, chunk_size=50, overlap=10):
    """Split text into overlapping word chunks for better retrieval coverage."""
    words = text.split()
    if not words:
        return []

    if len(words) <= chunk_size:
        return [" ".join(words)]

    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def build_idf(all_term_freqs):
    """Compute inverse document frequency across all chunks."""
    doc_count = len(all_term_freqs)
    if doc_count == 0:
        return {}

    doc_freq = {}
    for tf in all_term_freqs:
        for term in tf:
            doc_freq[term] = doc_freq.get(term, 0) + 1

    return {
        term: math.log((1 + doc_count) / (1 + df)) + 1
        for term, df in doc_freq.items()
    }


def tfidf_vector(term_freq, idf_map):
    """Sparse TF-IDF vector as {term: weight}."""
    total = sum(term_freq.values()) or 1
    return {
        term: (count / total) * idf_map.get(term, 0.0)
        for term, count in term_freq.items()
    }


def cosine_similarity(vec1, vec2):
    if not vec1 or not vec2:
        return 0.0

    common = set(vec1) & set(vec2)
    dot = sum(vec1[t] * vec2[t] for t in common)
    mag1 = math.sqrt(sum(v * v for v in vec1.values()))
    mag2 = math.sqrt(sum(v * v for v in vec2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def find_relevant_chunks(query, db_name=DB_NAME, top_k=TOP_K, min_similarity=MIN_SIMILARITY):
    """
    TF-IDF cosine search over all stored chunks.
    Returns list of dicts: content, similarity, source.
    """
    documents = fetch_all_documents(db_name)
    if not documents:
        return []

    idf_map = get_metadata("idf", db_name)
    if not idf_map:
        all_tfs = [doc["term_freq"] for doc in documents]
        idf_map = build_idf(all_tfs)

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    query_vec = tfidf_vector(term_frequencies(query_tokens), idf_map)
    threshold = min_similarity if len(query_tokens) > 1 else min_similarity * 0.5

    results = []
    for doc in documents:
        doc_vec = tfidf_vector(doc["term_freq"], idf_map)
        tfidf_score = cosine_similarity(query_vec, doc_vec)
        lexical_score = lexical_overlap_score(
            query_tokens, doc["content"], doc["term_freq"].keys()
        )
        similarity = combined_similarity(tfidf_score, lexical_score)
        if similarity >= threshold:
            results.append({
                "content": doc["content"],
                "similarity": similarity,
                "source": doc["source"],
            })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]
