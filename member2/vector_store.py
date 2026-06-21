import faiss
import json
import numpy as np
import os
from typing import List, Dict

from embeddings import SentenceTransformerEmbeddings


def build_and_persist_faiss_index(
    chunks: List[Dict],
    index_path: str = "faiss_index"
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

    texts = [c["text"] for c in chunks]

    embeddings = SentenceTransformerEmbeddings()
    vectors = np.array(embeddings.embed_documents(texts)).astype("float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    os.makedirs(index_path, exist_ok=True)

    faiss.write_index(index, os.path.join(index_path, "index.faiss"))

    # metadata MUST include text for retrieval
    meta = chunks

    with open(os.path.join(index_path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"FAISS index saved to {index_path}")


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