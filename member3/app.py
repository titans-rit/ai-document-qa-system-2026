"""
Intelligent Document Q&A Assistant — Member 3 starter skeleton
Run with: streamlit run app.py

This is a STANDALONE demo. The retrieve_chunks() function below is a
placeholder — once Member 2 builds the real embedding + FAISS/Chroma
retrieval pipeline, swap it in here without touching anything else.
"""

import streamlit as st
import ollama
import io
import re
try:
    import PyPDF2
    HAS_PYPDF2 = True
except Exception:
    HAS_PYPDF2 = False

st.set_page_config(page_title="Document Q&A Assistant", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------
# Theme — dark charcoal base, teal accent, monospace touches for
# citations (evokes log/config output, fitting for technical docs)
# ---------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #0F1115;
    --panel: #171A21;
    --accent: #4FD1C5;
    --text: #E7E5DF;
    --muted: #8B92A0;
}

[data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: var(--bg);
}

[data-testid="stSidebar"] {
    background-color: var(--panel);
    border-right: 1px solid #262B36;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text) !important;
}

h1 {
    letter-spacing: -0.02em;
}

p, span, label, div {
    font-family: 'Inter', sans-serif;
}

/* eyebrow line under the title */
.eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--accent);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

/* citation "hit" styling — terminal grep look */
.cite-hit {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    background: var(--panel);
    border-left: 3px solid var(--accent);
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.5rem;
    border-radius: 0 4px 4px 0;
    color: var(--text);
}

.cite-hit .src {
    color: var(--accent);
    font-weight: 500;
}

.cite-hit .body {
    color: var(--muted);
    margin-top: 0.3rem;
    display: block;
}

[data-testid="stChatMessage"] {
    background-color: var(--panel);
    border-radius: 10px;
    border: 1px solid #262B36;
}

button {
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="eyebrow">// internal · pdf-grounded retrieval</div>', unsafe_allow_html=True)
st.title("📄 Document Q&A Assistant")

# ---------------------------------------------------------------------
# Sidebar — uploaded files list + clear chat button
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs (user guides, release notes, SQA test cases, KCS articles)",
        type="pdf",
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) ready")
        for f in uploaded_files:
            st.caption(f"📄 {f.name}")
    else:
        st.info("No documents uploaded yet")

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------
# 2. PLACEHOLDER retrieval — replace this with Member 2's real function.
#    Contract: retrieve_chunks(question) -> list of dicts with
#    'text', 'source_file', 'page'
# ---------------------------------------------------------------------
def retrieve_chunks(question: str):
    # Simple local retrieval over uploaded PDFs.
    # Requires PyPDF2. If missing, notify via Streamlit error and return empty.
    if not HAS_PYPDF2:
        st.error("PyPDF2 is required for local PDF search. Install with `pip install PyPDF2`.")
        return []

    if not uploaded_files:
        return []

    query_tokens = set(w for w in re.findall(r"\w+", question.lower()) if len(w) > 2)
    hits = []

    for f in uploaded_files:
        try:
            data = f.read()
            reader = PyPDF2.PdfReader(io.BytesIO(data))
        except Exception:
            continue

        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""

            text_l = text.lower()
            score = sum(1 for t in query_tokens if t in text_l)
            if score > 0:
                hits.append({
                    "text": text.strip().replace("\n", " ")[:800],
                    "source_file": f.name,
                    "page": i + 1,
                    "score": score,
                })

    # If no hits found, fall back to returning the first page of the first PDF
    if not hits and uploaded_files:
        try:
            f = uploaded_files[0]
            data = f.read()
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            text = reader.pages[0].extract_text() or ""
            return [{"text": text.strip().replace("\n", " "), "source_file": f.name, "page": 1}]
        except Exception:
            return []

    # sort by score desc and return top 3 (strip score before returning)
    hits.sort(key=lambda x: x["score"], reverse=True)
    out = [{"text": h["text"], "source_file": h["source_file"], "page": h["page"]} for h in hits[:3]]
    return out

# ---------------------------------------------------------------------
# 3. LLM call — answers ONLY from retrieved chunks, with citations
# ---------------------------------------------------------------------
def ask_llm(question: str, chunks: list) -> str:
    context = "\n\n".join(
        f"[Source: {c['source_file']}, page {c['page']}]\n{c['text']}"
        for c in chunks
    )

    system_msg = (
        "You are a documentation assistant. Use ONLY the provided context to answer. "
        "Be concise and factual. Return the reply in Markdown with two sections: '### Answer' "
        "(a short, direct answer, max 150 words) and '### Sources' (a bullet list of file·page refs). "
        "If the answer is not present in the context, respond exactly: \"I couldn't find this in the uploaded documents.\""
    )

    user_msg = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        options={"temperature": 0.3},
    )

    return response["message"]["content"]

# ---------------------------------------------------------------------
# 4. Chat UI
# ---------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not uploaded_files:
            answer = "Please upload at least one PDF before asking a question."
            st.warning(answer)
        else:
            with st.spinner("Searching documents and generating answer..."):
                try:
                    chunks = retrieve_chunks(question)
                    answer = ask_llm(question, chunks)
                    st.markdown(answer)

                    with st.expander("Sources used"):
                        for c in chunks:
                            st.markdown(
                                f'<div class="cite-hit">'
                                f'<span class="src">{c["source_file"]} · page {c["page"]}</span>'
                                f'<span class="body">{c["text"]}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                except Exception as e:
                    answer = (
                        "Something went wrong while generating the answer. "
                        "Make sure the Ollama app is running in the background."
                    )
                    st.error(answer)
                    st.caption(f"Technical details: {e}")

    st.session_state.messages.append({"role": "assistant", "content": answer})