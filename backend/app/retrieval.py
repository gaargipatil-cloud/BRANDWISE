"""Passage chunking and TF-IDF semantic retrieval over Finwise brand guidelines."""
import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BRAND_PACK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "brand_pack"))

_PASSAGES = []
_VECTORIZER = None
_TFIDF_MATRIX = None


def init_brand_corpus():
    """Read markdown and json files from data/brand_pack, chunking into numbered passages."""
    global _PASSAGES, _VECTORIZER, _TFIDF_MATRIX
    passages = []

    files = glob.glob(os.path.join(BRAND_PACK_DIR, "*.md"))
    for file_path in files:
        filename = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        blocks = content.split("\n\n")
        chunk_idx = 1
        for block in blocks:
            clean_block = block.strip()
            if len(clean_block) > 20:
                passages.append({
                    "id": f"{filename}#p{chunk_idx}",
                    "source_name": filename,
                    "text": clean_block
                })
                chunk_idx += 1

    _PASSAGES = passages
    corpus_texts = [p["text"] for p in _PASSAGES]

    if corpus_texts:
        _VECTORIZER = TfidfVectorizer(stop_words="english")
        _TFIDF_MATRIX = _VECTORIZER.fit_transform(corpus_texts)


def retrieve_top_passages(query: str, top_k: int = 3) -> list:
    """Return the top k relevant brand passages using TF-IDF cosine similarity."""
    global _PASSAGES, _VECTORIZER, _TFIDF_MATRIX
    if not _PASSAGES or _VECTORIZER is None or _TFIDF_MATRIX is None:
        init_brand_corpus()

    if not _PASSAGES or not query.strip():
        return []

    query_vec = _VECTORIZER.transform([query])
    similarities = cosine_similarity(query_vec, _TFIDF_MATRIX).flatten()

    top_indices = similarities.argsort()[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "id": _PASSAGES[idx]["id"],
            "source_name": _PASSAGES[idx]["source_name"],
            "text": _PASSAGES[idx]["text"],
            "score": float(similarities[idx])
        })
    return results


def get_all_passages_text() -> str:
    """Return consolidated text of all loaded brand pack passages."""
    global _PASSAGES
    if not _PASSAGES:
        init_brand_corpus()
    return "\n\n".join([f"[{p['id']}] {p['text']}" for p in _PASSAGES])