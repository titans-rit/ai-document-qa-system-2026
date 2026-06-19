from typing import List

from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings:
	"""Lightweight wrapper around SentenceTransformer to match LangChain's
	embeddings interface used in this project.
	"""

	def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
		self.model_name = model_name
		self.model = SentenceTransformer(model_name)

	def embed_documents(self, texts: List[str]) -> List[List[float]]:
		return [vec.tolist() for vec in self.model.encode(texts, show_progress_bar=False)]

	def embed_query(self, text: str) -> List[float]:
		vec = self.model.encode([text], show_progress_bar=False)
		return vec[0].tolist()


def embed_text(text: str, model_name: str = "all-MiniLM-L6-v2") -> List[float]:
	"""Convenience function to get a single text embedding."""
	emb = SentenceTransformer(model_name)
	vec = emb.encode([text], show_progress_bar=False)
	return vec[0].tolist()


if __name__ == "__main__":
	sample = "Hello Timer triggers every 100 ms"
	print("Loading model and embedding sample text...")
	vector = embed_text(sample)
	print("Sample text:", sample)
	print("Embedding length:", len(vector))
