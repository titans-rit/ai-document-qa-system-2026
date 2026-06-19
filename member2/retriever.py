from typing import List

from member2.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS


def retrieve(query: str, index_path: str = "faiss_index", k: int = 3):
	embeddings = SentenceTransformerEmbeddings()

	print(f"Loading FAISS index from {index_path}/")
	try:
		docsearch = FAISS.load_local(index_path, embeddings)
	except Exception:
		raise RuntimeError("Failed to load FAISS index. Make sure it exists and LangChain supports load_local.")

	print(f"Searching for: {query}")
	results = docsearch.similarity_search(query, k=k)

	output = []
	for doc in results:
		metadata = doc.metadata or {}
		output.append({
			"text": doc.page_content,
			"source_file": metadata.get("source_file"),
			"page": metadata.get("page"),
		})

	return output


if __name__ == "__main__":
	q = "What is Hello Timer?"
	res = retrieve(q)
	for i, item in enumerate(res, 1):
		print(f"Result {i}:")
		print(item["text"]) 
		print("Source:", item.get("source_file"), "Page", item.get("page"))
		print("---")
