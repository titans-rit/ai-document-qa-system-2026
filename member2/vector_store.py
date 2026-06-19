import os
from typing import List, Dict

from langchain.schema import Document
from langchain.vectorstores import FAISS

from member2.embeddings import SentenceTransformerEmbeddings


def build_and_persist_faiss_index(index_path: str = "faiss_index") -> FAISS:
	# Dummy chunks as requested
	chunks: List[Dict] = [
		{"text": "Hello Timer triggers every 100 ms", "page": 10, "source_file": "SFS.pdf"},
		{"text": "Hello Timer validation ensures timing accuracy", "page": 12, "source_file": "SFS.pdf"},
		{"text": "Hello Timer can be stopped and restarted programmatically", "page": 15, "source_file": "SFS.pdf"},
	]

	documents = [
		Document(page_content=item["text"], metadata={"page": item["page"], "source_file": item["source_file"]})
		for item in chunks
	]

	embeddings = SentenceTransformerEmbeddings()

	print("Building FAISS index from documents...")
	faiss_store = FAISS.from_documents(documents, embeddings)

	# Ensure index directory exists
	os.makedirs(index_path, exist_ok=True)
	print(f"Saving FAISS index to {index_path}/")
	try:
		faiss_store.save_local(index_path)
	except Exception:
		# Some LangChain versions use "persist" or different APIs; tolerate failures
		print("Warning: save_local failed — check LangChain version. Index exists in memory.")

	return faiss_store


if __name__ == "__main__":
	store = build_and_persist_faiss_index()
	print("Index built. Example doc count:", len(store.docstore._dict))
