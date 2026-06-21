import os
from typing import List

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, EMBEDDING_DEVICE


class SentenceTransformerEmbeddings:
    """Wrapper for embedding generation (FAISS + retrieval use)"""

    def __init__(self, model_name: str = None, device: str = None):
        model_name = model_name or EMBEDDING_MODEL
        device = device or EMBEDDING_DEVICE
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], show_progress_bar=False)[0].tolist()


def embed_text(text: str) -> List[float]:
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model.encode([text], show_progress_bar=False)[0].tolist()


if __name__ == "__main__":
    sample = "Hello Timer triggers every 100 ms"
    print(embed_text(sample))