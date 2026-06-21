import faiss
import json
import numpy as np
import os
from typing import List, Dict

from member2.embeddings import SentenceTransformerEmbeddings
from config import FAISS_INDEX_DIR, CHUNKS_JSON_PATH


def build_and_persist_faiss_index(
    chunks: List[Dict],
    index_path: str = None
) -> None:
    """
    INPUT FORMAT (FROM MEMBER 1):
    [
        {
            "text": "...",
            "page": 10,
            "source_file": "file.pdf"
        }
    ]
    """

    index_path = index_path or FAISS_INDEX_DIR

    texts = [c["text"] for c in chunks]

    embeddings = SentenceTransformerEmbeddings()
    vectors = np.array(embeddings.embed_documents(texts)).astype("float32")

    if vectors.size == 0:
        raise ValueError("No vectors were generated from chunks")

    # Normalize vectors for cosine similarity (use IndexFlatIP)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    os.makedirs(index_path, exist_ok=True)

    faiss.write_index(index, os.path.join(index_path, "index.faiss"))

    # metadata MUST include text for retrieval
    meta = chunks

    with open(os.path.join(index_path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"FAISS index saved to {index_path}")


def build_from_chunks_json(chunks_json_path: str = None, index_path: str = None) -> None:
    """Helper to build index directly from Member1's chunks.json file."""
    chunks_json_path = chunks_json_path or CHUNKS_JSON_PATH
    index_path = index_path or FAISS_INDEX_DIR

    if not os.path.exists(chunks_json_path):
        raise FileNotFoundError(f"Chunks JSON not found: {chunks_json_path}")

    with open(chunks_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    build_and_persist_faiss_index(chunks, index_path=index_path)


if __name__ == "__main__":
    # test sample only
    sample_chunks = [
        {
            "text": "Hello Timer triggers every 100 ms",
            "page": 10,
            "source_file": "SFS.pdf"
        }
    ]
    build_and_persist_faiss_index(sample_chunks)