import os
import re
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# ── Docling ───────────────────────────────────────────────────────────────────
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# ── LangChain splitters ───────────────────────────────────────────────────────
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_core.documents import Document

# ── Language detection ────────────────────────────────────────────────────────
try:
    from langdetect import detect as _detect_lang
    _LANGDETECT_OK = True
except ImportError:
    _LANGDETECT_OK = False

# ── Shared config ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    SUPPORTED_EXTENSIONS, DOC_TYPE_MAP, CHUNK_CONFIG, CHUNKS_JSON_PATH
)

log = logging.getLogger(__name__)

# Markdown headers to split on during chunking
_MD_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]

# Text noise patterns to strip after extraction
_NOISE_RE = [
    (re.compile(r"Page\s+\d+\s+of\s+\d+", re.I), ""),
    (re.compile(r"©\s*\d{4}[^\n]*",        re.I), ""),
    (re.compile(r"Confidential[^\n]*",      re.I), ""),
    (re.compile(r"Proprietary[^\n]*",       re.I), ""),
    (re.compile(r"\ufb01"),                        "fi"),
    (re.compile(r"\ufb02"),                        "fl"),
    (re.compile(r"[\u2018\u2019]"),                "'"),
    (re.compile(r"[\u201c\u201d]"),                '"'),
    (re.compile(r"[\u2013\u2014]"),                "-"),
    (re.compile(r"\u00a0"),                        " "),
    (re.compile(r"\x00"),                          ""),
    (re.compile(r"\n{3,}"),                        "\n\n"),
    (re.compile(r"[ \t]{2,}"),                     " "),
]


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def detect_doc_type(filename: str) -> str:
    """Classify document type from filename keywords."""
    name = filename.lower()
    for dtype, keywords in DOC_TYPE_MAP.items():
        if any(k in name for k in keywords):
            return dtype
    return "unknown"


def detect_language(text: str) -> str:
    if not _LANGDETECT_OK or len(text.strip()) < 40:
        return "unknown"
    try:
        return _detect_lang(text)
    except Exception:
        return "unknown"


def chunk_id(text: str, source: str, idx: int) -> str:
    """Stable deterministic MD5 chunk ID."""
    return hashlib.md5(f"{source}::{idx}::{text[:100]}".encode()).hexdigest()[:14]


def clean_text(text: str) -> str:
    """Remove noise while preserving Docling markdown structure."""
    for pattern, replacement in _NOISE_RE:
        text = pattern.sub(replacement, text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — EXTRACTION  (Docling)
# ══════════════════════════════════════════════════════════════════════════════

def build_converter() -> DocumentConverter:
    """
    Build Docling converter with:
     - TableFormer ML  → accurate table recognition
     - EasyOCR         → scanned PDF support
    """
    pdf_opts = PdfPipelineOptions(do_table_structure=True, do_ocr=True)
    pdf_opts.table_structure_options.do_cell_matching = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
    )


def extract_file(filepath: str, converter: DocumentConverter) -> dict:
    """
    Extract a single document → structured markdown + tables.

    Returns:
        {
          "markdown": str,          # full doc as structured markdown
          "tables":   list[dict],   # each table as plain text
          "metadata": dict,         # source, doc_type, headings, etc.
        }
    """
    p        = Path(filepath)
    filename = p.name
    doc_type = detect_doc_type(filename)

    log.info(f"Extracting [{doc_type.upper()}] {filename}")
    result = converter.convert(str(p))
    doc    = result.document

    markdown = clean_text(doc.export_to_markdown())

    # Extract tables as plain text strings
    tables = []
    for i, tbl in enumerate(doc.tables):
        try:
            text = tbl.export_to_dataframe().to_string(index=False)
        except Exception:
            text = tbl.export_to_markdown()
        tables.append({"table_index": i, "text": clean_text(f"[TABLE {i+1}]\n{text}")})

    # Collect heading list for metadata
    headings = [
        item.text for item in doc.texts
        if hasattr(item, "label") and "heading" in str(item.label).lower()
    ][:10]

    log.info(f"  ✓ {filename} — {len(markdown.split())} words, {len(tables)} tables")

    return {
        "markdown": markdown,
        "tables":   tables,
        "metadata": {
            "source":       filename,
            "filepath":     str(p.resolve()),
            "doc_type":     doc_type,
            "headings":     headings,
            "table_count":  len(tables),
            "extracted_at": datetime.now().isoformat(),
        },
    }


def extract_all(input_path: str, converter: DocumentConverter) -> list[dict]:
    """Extract all supported documents from a file or directory."""
    p = Path(input_path)
    files = (
        [p] if p.is_file()
        else [f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    )
    if not files:
        log.warning(f"No supported files found in: {input_path}")
        return []

    log.info(f"Found {len(files)} document(s)")
    results = []
    for f in files:
        try:
            results.append(extract_file(str(f), converter))
        except Exception as e:
            log.error(f"  ✗ Failed: {f.name} — {e}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def chunk_extraction(extracted: dict) -> list[Document]:
    """
    Two-pass chunking:
      Pass 1 — MarkdownHeaderTextSplitter  (split at section boundaries)
      Pass 2 — RecursiveCharacterTextSplitter  (enforce token size limit)
    Tables are kept as individual chunks.

    Returns LangChain Document objects ready for FAISS embedding.
    """
    meta     = extracted["metadata"]
    cfg      = CHUNK_CONFIG.get(meta["doc_type"], CHUNK_CONFIG["unknown"])
    docs     = []
    seen: set[str] = set()

    # Pass 1: split by markdown headings
    try:
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_MD_HEADERS, strip_headers=False
        )
        header_chunks = header_splitter.split_text(extracted["markdown"])
    except Exception:
        header_chunks = [Document(page_content=extracted["markdown"], metadata={})]

    # Pass 2: enforce size limit within each section
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )

    for h_chunk in header_chunks:
        for text in char_splitter.split_text(h_chunk.page_content):
            text = text.strip()
            if len(text.split()) < 8:
                continue
            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)

            idx = len(docs)
            section = (
                h_chunk.metadata.get("h1")
                or h_chunk.metadata.get("h2")
                or h_chunk.metadata.get("h3", "")
            )
            docs.append(Document(
                page_content=text,
                metadata={
                    "chunk_id":    chunk_id(text, meta["source"], idx),
                    "source":      meta["source"],
                    "filepath":    meta["filepath"],
                    "doc_type":    meta["doc_type"],
                    "section":     section,
                    "language":    detect_language(text),
                    "chunk_index": idx,
                    "word_count":  len(text.split()),
                    "is_table":    False,
                    "processed_at": meta["extracted_at"],
                }
            ))

    # Tables: one chunk each
    for tbl in extracted["tables"]:
        text = tbl["text"].strip()
        if not text or len(text.split()) < 4:
            continue
        h = hashlib.md5(text.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        idx = len(docs)
        docs.append(Document(
            page_content=text,
            metadata={
                "chunk_id":    chunk_id(text, meta["source"], idx),
                "source":      meta["source"],
                "filepath":    meta["filepath"],
                "doc_type":    meta["doc_type"],
                "section":     f"Table {tbl['table_index'] + 1}",
                "language":    detect_language(text),
                "chunk_index": idx,
                "word_count":  len(text.split()),
                "is_table":    True,
                "processed_at": meta["extracted_at"],
            }
        ))

    log.info(f"  Chunked {meta['source']} → {len(docs)} chunks")
    return docs


def chunk_all_extractions(extractions: list[dict]) -> list[Document]:
    """Chunk all extracted documents."""
    all_docs: list[Document] = []
    for ex in extractions:
        all_docs.extend(chunk_extraction(ex))
    log.info(f"Total chunks: {len(all_docs)}")
    return all_docs


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE  (handoff to Member 2)
#
#  ✅ CHANGED: simplified output — plain list of {text, page, source_file}
#     Member 2 only needs these three fields to build the vector store.
#     All internal processing metadata stays internal to Member 1.
# ══════════════════════════════════════════════════════════════════════════════

def save_chunks_json(documents: list[Document], path: str = CHUNKS_JSON_PATH) -> None:
    """
    Persist chunks as JSON for Member 2's embedding pipeline.

    Output format (simplified — agreed team contract):
    [
        {
            "text":        "chunk text content ...",
            "page":        10,
            "source_file": "user_guide.pdf"
        },
        ...
    ]
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # ── CHANGED: flat list only — no "summary" wrapper, no internal metadata ──
    payload = [
        {
            "text":        d.page_content,
            "page":        d.metadata.get("chunk_index", 0),
            "source_file": d.metadata.get("source", ""),
        }
        for d in documents
    ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    log.info(f"Chunks saved → {path}  ({len(payload)} chunks)")


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API  (called by Member 3 Streamlit app)
# ══════════════════════════════════════════════════════════════════════════════

def process_documents(file_paths: list[str]) -> list[Document]:
    """
    Full Member 1 pipeline for a list of file paths.
    Used by the Streamlit app (Member 3) after upload.

    Returns list[Document] ready for Member 2's build_vector_store().
    """
    converter  = build_converter()
    extractions = []
    for fp in file_paths:
        try:
            extractions.append(extract_file(fp, converter))
        except Exception as e:
            log.error(f"Failed to extract {fp}: {e}")

    if not extractions:
        return []

    documents = chunk_all_extractions(extractions)
    save_chunks_json(documents)
    return documents


# ══════════════════════════════════════════════════════════════════════════════
#  CLI  (standalone testing)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Member 1 — Extract & Chunk Documents")
    parser.add_argument("--input", "-i", required=True, help="File or directory of documents")
    args = parser.parse_args()

    docs = process_documents(
        [str(f) for f in Path(args.input).rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        if Path(args.input).is_dir()
        else [args.input]
    )
    print(f"\n✅ Member 1 complete — {len(docs)} chunks ready in {CHUNKS_JSON_PATH}")