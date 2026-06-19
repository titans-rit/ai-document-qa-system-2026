from typing import List
import os
import sys
import json
import faiss
import numpy as np

# ensure local imports work when running script directly
sys.path.insert(0, os.path.dirname(__file__))

from embeddings import SentenceTransformerEmbeddings


def retrieve(query: str, index_path: str = "faiss_index", k: int = 3):
	idx_file = os.path.join(index_path, "index.faiss")
	meta_file = os.path.join(index_path, "metadata.json")

	if not os.path.exists(idx_file) or not os.path.exists(meta_file):
		raise FileNotFoundError("Index or metadata not found. Run vector_store.py to build the index first.")

	index = faiss.read_index(idx_file)
	with open(meta_file, "r", encoding="utf-8") as f:
		metadata = json.load(f)

	embeddings = SentenceTransformerEmbeddings()
	q_vec = np.array(embeddings.embed_query(query), dtype="float32").reshape(1, -1)

	D, I = index.search(q_vec, k)

	output = []
	for idx in I[0]:
		if idx < 0 or idx >= len(metadata):
			continue
		item = metadata[idx]
		output.append({"text": item.get("text"), "source_file": item.get("source_file"), "page": item.get("page")})

	return output


if __name__ == "__main__":
	q = "What is Hello Timer?"
	res = retrieve(q)
	for i, item in enumerate(res, 1):
		print(f"Result {i}:")
		print(item["text"]) 
		print("Source:", item.get("source_file"), "Page", item.get("page"))
		print("---")
