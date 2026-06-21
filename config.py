"""
config.py — Shared configuration for all 3 members
All settings in one place. Change here, applies everywhere.
"""

# ── Paths ────────────────────────────────────────────────────────────────────
UPLOAD_DIR       = "./uploaded_docs"        # where Streamlit saves uploads
CHUNKS_JSON_PATH = "./data/chunks.json"     # Member1 → Member2 handoff
FAISS_INDEX_DIR  = "./data/faiss_index"     # Member2 saves, Member3 loads

# ── Docling / Extraction ─────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".htm"}

DOC_TYPE_MAP = {
    "user_guide":   ["user_guide", "userguide", "manual", "handbook"],
    "release_note": ["release", "changelog", "relnote", "release_note"],
    "sqa":          ["sqa", "test_case", "testcase", "test_plan"],
    "kcs":          ["kcs", "article", "kb_", "knowledge"],
}

# ── Chunking — tuned per document type ───────────────────────────────────────
CHUNK_CONFIG = {
    "user_guide":   {"chunk_size": 600, "chunk_overlap": 80},
    "release_note": {"chunk_size": 450, "chunk_overlap": 60},
    "sqa":          {"chunk_size": 350, "chunk_overlap": 40},
    "kcs":          {"chunk_size": 600, "chunk_overlap": 80},
    "unknown":      {"chunk_size": 512, "chunk_overlap": 64},
}

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"          # change to "cuda" if GPU available

# ── FAISS retrieval ───────────────────────────────────────────────────────────
TOP_K_RESULTS = 5                 # number of chunks to retrieve per query
MIN_SIMILARITY = 0.25            # minimum cosine similarity to consider a chunk relevant (0-1)

# ── Ollama / LLM ─────────────────────────────────────────────────────────────
OLLAMA_MODEL    = "llama3"
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_TEMPERATURE = 0.1
LLM_CONTEXT_WINDOW = 4096

# ── RAG Prompt ────────────────────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """You are an expert technical assistant for enterprise documents.
Use ONLY the context provided below to answer the question accurately.
If the answer is not found in the context, respond with the exact phrase:
"I couldn't find this in the uploaded documents."
Always cite the source document name and page number in your answer.

Context:
{context}

Question: {question}

Answer (include source file and page number):"""