import faiss
import json
import numpy as np
import os
import sys
from typing import List, Dict

# ensure local imports work when running script directly
sys.path.insert(0, os.path.dirname(__file__))

from embeddings import SentenceTransformerEmbeddings


def build_and_persist_faiss_index(index_path: str = "faiss_index") -> None:
	# Dummy chunks as requested
	chunks: List[Dict] = [
		{"text": "Hello Timer triggers every 100 ms", "page": 10, "source_file": "SFS.pdf"},
		{"text": "Hello Timer validation ensures timing accuracy", "page": 12, "source_file": "SFS.pdf"},
		{"text": "Hello Timer can be stopped and restarted programmatically", "page": 15, "source_file": "SFS.pdf"},
	]

	texts = [item["text"] for item in chunks]
	metadatas = [{"page": item["page"], "source_file": item["source_file"]} for item in chunks]

	embeddings = SentenceTransformerEmbeddings()
	print("Computing embeddings for texts...")
	vectors = np.array(embeddings.embed_documents(texts)).astype("float32")

	dim = vectors.shape[1]
	index = faiss.IndexFlatL2(dim)
	index.add(vectors)

	os.makedirs(index_path, exist_ok=True)
	faiss.write_index(index, os.path.join(index_path, "index.faiss"))

	# save metadata (texts + metadatas) to JSON
	meta = [{"text": t, **m} for t, m in zip(texts, metadatas)]
	with open(os.path.join(index_path, "metadata.json"), "w", encoding="utf-8") as f:
		json.dump(meta, f, ensure_ascii=False, indent=2)

	print(f"Saved FAISS index and metadata to {index_path}/")


if __name__ == "__main__":
	build_and_persist_faiss_index()
	print("Index built.")
