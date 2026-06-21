import os
import json
import faiss
import numpy as np
from member2.embeddings import SentenceTransformerEmbeddings
from config import FAISS_INDEX_DIR, TOP_K_RESULTS, MIN_SIMILARITY


def retrieve(query: str, index_path: str = None, k: int = None):
    """Return list of {'text','source_file','page'} for top-k similar chunks."""

    index_path = index_path or FAISS_INDEX_DIR
    k = k or TOP_K_RESULTS

    idx_file = os.path.join(index_path, "index.faiss")
    meta_file = os.path.join(index_path, "metadata.json")

    if not os.path.exists(idx_file) or not os.path.exists(meta_file):
        raise FileNotFoundError("Run vector_store.build_and_persist_faiss_index() first")

    index = faiss.read_index(idx_file)

    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    embeddings = SentenceTransformerEmbeddings()
    q_vec = np.array(
        embeddings.embed_query(query),
        dtype="float32"
    ).reshape(1, -1)

    # normalize query vector for cosine similarity
    q_norm = np.linalg.norm(q_vec)
    if q_norm == 0:
        q_norm = 1.0
    q_vec = q_vec / q_norm

    D, I = index.search(q_vec, k)

    output = []

    # D contains inner-product scores (cosine since vectors normalized)
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata):
            continue

        # filter by MIN_SIMILARITY to improve precision
        if score < MIN_SIMILARITY:
            continue

        item = metadata[idx]

        output.append({
            "text": item.get("text"),
            "source_file": item.get("source_file"),
            "page": item.get("page"),
            "score": float(score),
        })

    # sort by score desc
    output.sort(key=lambda x: x.get("score", 0), reverse=True)

    return output


if __name__ == "__main__":
    q = "What is Hello Timer?"
    res = retrieve(q)
    print(json.dumps(res, indent=2))