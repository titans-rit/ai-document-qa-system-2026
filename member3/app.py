"""
Intelligent Document Q&A Assistant — Member 3 UI (modernized)
Run with: streamlit run member3/app.py

This file contains only UI/UX improvements. Backend logic (extraction,
embeddings, FAISS, retrieval, and LLM calling) is reused unchanged.
"""
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
try:
    import ollama
    OLLAMA_PY_AVAILABLE = True
except Exception:
    ollama = None
    OLLAMA_PY_AVAILABLE = False
import requests
from member2.retriever import retrieve
from member1.extractor import process_documents
from member2.vector_store import build_from_chunks_json
from config import (
    UPLOAD_DIR,
    CHUNKS_JSON_PATH,
    FAISS_INDEX_DIR,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    LLM_TEMPERATURE,
    RAG_PROMPT_TEMPLATE,
    MIN_SIMILARITY,
    TOP_K_RESULTS,
)

st.set_page_config(page_title="Document Q&A Assistant", page_icon="📄", layout='wide')

# ---------------------------------------------------------------------
# Backend adapters (kept unchanged semantically)
# ---------------------------------------------------------------------

def retrieve_chunks(question: str):
    return retrieve(question)


def ask_llm(question: str, chunks: list) -> str:
    # Select relevant chunks
    filtered = [c for c in chunks if c.get("score", 1.0) >= MIN_SIMILARITY]
    if not filtered:
        filtered = chunks[:TOP_K_RESULTS]

    # Build context
    context_parts = []
    total_chars = 0
    for c in filtered:
        part = f"[Source: {c['source_file']}, page {c['page']}]\n{c['text']}"
        context_parts.append(part)
        total_chars += len(part)
        if total_chars > 3000:
            break

    context = "\n\n".join(context_parts)
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    # Try Python client then HTTP fallback; capture response text for post-checks
    resp_text = None
    try:
        if OLLAMA_PY_AVAILABLE and ollama is not None:
            try:
                ollama.api_base = OLLAMA_BASE_URL
            except Exception:
                pass
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_TEMPERATURE,
            )
            if isinstance(response, dict):
                resp_text = (
                    response.get("message", {}).get("content")
                    or (response.get("choices", [{}])[0].get("message", {}).get("content"))
                    or response.get("choices", [{}])[0].get("text")
                    or ""
                )
            else:
                resp_text = str(response)
        else:
            url = OLLAMA_BASE_URL.rstrip("/") + "/v1/chat"
            payload = {
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": LLM_TEMPERATURE,
            }
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            resp_text = (
                data.get("message", {}).get("content")
                or (data.get("choices", [{}])[0].get("message", {}).get("content"))
                or data.get("choices", [{}])[0].get("text")
                or ""
            )
    except Exception:
        # Deterministic fallback - return authoritative excerpts
        if not chunks:
            return "I couldn't find this in the uploaded documents."
        top_chunks = (filtered or chunks)[:3]
        snippets = []
        for t in top_chunks:
            s = t.get("text", "").strip().replace("\n", " ")
            if len(s) > 400:
                s = s[:400].rsplit(" ", 1)[0] + "..."
            snippets.append(f"{s} (Source: {t.get('source_file')}, page {t.get('page')})")
        return "\n\n".join(snippets)

    # Post-process LLM output: enforce that answer cites uploaded documents only.
    if not resp_text or not resp_text.strip():
        # empty response -> fallback
        if not chunks:
            return "I couldn't find this in the uploaded documents."
        top_chunks = (filtered or chunks)[:3]
        snippets = []
        for t in top_chunks:
            s = t.get("text", "").strip().replace("\n", " ")
            if len(s) > 400:
                s = s[:400].rsplit(" ", 1)[0] + "..."
            snippets.append(f"{s} (Source: {t.get('source_file')}, page {t.get('page')})")
        return "\n\n".join(snippets)

    resp_lower = resp_text.lower()
    refusal_phrase = "i couldn't find this in the uploaded documents."
    if refusal_phrase in resp_lower:
        return resp_text

    # Check for explicit source citations (file names or the literal 'Source:')
    source_names = set([str(c.get('source_file', '')).lower() for c in chunks])
    cited = False
    if "source:" in resp_text.lower():
        cited = True
    else:
        for name in source_names:
            if name and name in resp_lower:
                cited = True
                break

    if not cited:
        # LLM didn't cite sources explicitly; return deterministic excerpts instead
        if not chunks:
            return "I couldn't find this in the uploaded documents."
        top_chunks = (filtered or chunks)[:3]
        snippets = []
        for t in top_chunks:
            s = t.get("text", "").strip().replace("\n", " ")
            if len(s) > 400:
                s = s[:400].rsplit(" ", 1)[0] + "..."
            snippets.append(f"{s} (Source: {t.get('source_file')}, page {t.get('page')})")
        return "\n\n".join(snippets)

    return resp_text



# ---------------------------------------------------------------------
# UI / UX (only changes below)
# ---------------------------------------------------------------------

# CSS and styling
CSS = '''
/* Blue & White Enterprise Theme */
:root{
    --bg:#f8fbff; --card:#ffffff; --muted:#6b7280; --accent:#0b5cff; --accent-2:#2563eb; --glass:rgba(11,92,255,0.06); --success:#16a34a; --danger:#dc2626; --border: #e6eef8;
}
body { background: var(--bg); }
.app-wrapper{font-family: Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; padding:18px}
.top-nav{display:flex;align-items:center;justify-content:space-between;padding:14px 8px;margin-bottom:8px}
.logo{display:flex;align-items:center;gap:12px}
.logo-mark{width:44px;height:44px;border-radius:10px;background:linear-gradient(135deg,var(--accent),var(--accent-2));color:white;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:16px}
.logo-text{font-weight:700;color:#0f172a}
.status-badge{background:linear-gradient(90deg,#e6f0ff,#f0f7ff);padding:8px 12px;border-radius:999px;color:var(--accent-2);font-weight:600;border:1px solid var(--border)}

.side-card{background:var(--card);border-radius:12px;padding:16px;border:1px solid var(--border);box-shadow:0 6px 18px rgba(12,37,75,0.04)}
.side-title{font-weight:700;color:#0f172a;margin-bottom:8px}
.empty-upload{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:18px;border-radius:10px;border:1px dashed rgba(11,92,255,0.12);background:linear-gradient(180deg, rgba(11,92,255,0.02), rgba(255,255,255,0.0));text-align:center}
.doc-list{display:flex;flex-direction:column;gap:8px}
.doc-card{display:flex;align-items:center;gap:10px;padding:8px;border-radius:8px;border:1px solid var(--border)}
.doc-icon{font-size:18px}
.doc-name{font-size:13px;color:#0f172a}
.answer-content,.bubble,.doc-name,.doc-card{font-family: Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;}

.col-left-card{background:linear-gradient(180deg, #f2f8ff, #ffffff)}
.col-main-card{background:var(--card)}
.col-right-card{background:linear-gradient(180deg,#f8fbff,#ffffff)}

.stat-widgets{display:flex;gap:10px;margin-top:8px}
.stat{background:linear-gradient(180deg,#ffffff,#fbfdff);padding:10px;border-radius:10px;border:1px solid var(--border);flex:1;text-align:center}
.stat-num{font-size:18px;font-weight:800;color:var(--accent-2)}
.stat-label{font-size:12px;color:var(--muted);margin-top:4px}

.main-card{background:var(--card);border-radius:12px;padding:18px;border:1px solid var(--border);box-shadow:0 10px 30px rgba(12,37,75,0.04)}
.welcome h2{margin:0 0 6px 0}
.lead{color:var(--muted);margin:0 0 12px 0}
.chat-area{max-height:56vh;overflow:auto;padding:8px;display:flex;flex-direction:column;gap:10px}
.msg{display:flex}
.msg.user{justify-content:flex-end}
.bubble{display:inline-block;padding:10px 14px;border-radius:14px;max-width:78%;font-size:14px}
.user-bubble{background:linear-gradient(90deg,#eef2ff,#f8fbff);color:#0f172a;border:1px solid rgba(37,99,235,0.08)}
.assistant-card{background:linear-gradient(180deg,var(--glass), rgba(11,92,255,0.02));border-radius:12px;padding:12px;margin-bottom:6px;border:1px solid rgba(11,92,255,0.06)}
.assistant-meta{font-size:12px;color:var(--muted);margin-bottom:8px}
.answer-content{font-size:14px;color:#0f172a;line-height:1.6}
.source-badge{display:inline-block;background:#eef6ff;color:var(--accent-2);padding:6px 10px;border-radius:999px;font-size:12px;margin-right:6px}

.send-row{display:flex;gap:8px;margin-top:12px}
.send-input{flex:1}
.send-btn{background:var(--accent);color:white;padding:10px 14px;border-radius:10px;border:none}

@media (max-width: 900px){
    .stat-widgets{flex-direction:column}
}
'''

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

# Initialize session
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# Top navigation
st.markdown("""
<div class='top-nav'>
  <div class='nav-left'>
    <div class='logo'>
      <div class='logo-mark'>AI</div>
      <div class='logo-text'>Intelligent Document Assistant</div>
    </div>
  </div>
  <div class='nav-right'>
    <div class='status-badge'>Ready</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Main three-column layout: left (upload + stats), center (chat), right (previous chats)
col_left, col_main, col_right = st.columns([1, 2.2, 0.9])

with col_left:
    st.markdown("<div class='side-card'>", unsafe_allow_html=True)
    st.markdown("<div class='side-title'>Upload Documents</div>", unsafe_allow_html=True)
    # Drag and drop upload area
    uploaded_files = st.file_uploader("\n", type=["pdf"], accept_multiple_files=True, key="uploader_main")
    if uploaded_files:
        st.markdown(f"<div class='doc-list'>", unsafe_allow_html=True)
        for f in uploaded_files:
            st.markdown(f"<div class='doc-card'><div class='doc-icon'>📄</div><div class='doc-name'>{f.name}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Empty state for upload
        st.markdown('''
<div class='empty-upload'>
  <svg width="120" height="80" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 3v9" stroke="#2563EB" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M8 7l4-4 4 4" stroke="#2563EB" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M21 15v2a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-2" stroke="#93C5FD" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>
  <div style="margin-top:8px;font-weight:600;color:#0f172a">Upload Documents to Begin</div>
  <div style="font-size:12px;color:#6b7280">PDFs: user guides, release notes, test cases, KCS</div>
</div>
''', unsafe_allow_html=True)

    st.markdown("<div class='side-title' style='margin-top:12px'>Document Statistics</div>", unsafe_allow_html=True)
    try:
        uploaded_count = len(os.listdir(UPLOAD_DIR)) if os.path.isdir(UPLOAD_DIR) else (len(uploaded_files) if uploaded_files else 0)
    except Exception:
        uploaded_count = 0
    try:
        if os.path.exists(CHUNKS_JSON_PATH):
            with open(CHUNKS_JSON_PATH, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            chunks_count = len(data)
        else:
            chunks_count = 0
    except Exception:
        chunks_count = 0

    st.markdown(f"<div class='stat-widgets'><div class='stat'><div class='stat-num'>{uploaded_count}</div><div class='stat-label'>Documents</div></div><div class='stat'><div class='stat-num'>{chunks_count}</div><div class='stat-label'>Indexed Chunks</div></div><div class='stat'><div class='stat-num'>{OLLAMA_MODEL}</div><div class='stat-label'>Model</div></div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_main:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='welcome'><h2>Welcome to the Intelligent Document Assistant</h2><p class='lead'>Upload your documents on the left and ask questions here. Answers will reference uploaded documents only.</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='chat-area'>", unsafe_allow_html=True)
    # Render messages
    if not st.session_state.messages:
        st.markdown("<div class='empty-chat'>No conversation yet. Start by asking a question.</div>", unsafe_allow_html=True)
    for msg in st.session_state.messages:
        role = msg.get('role')
        content = msg.get('content')
        if role == 'user':
            st.markdown(f"<div class='msg user'><div class='bubble user-bubble'>{st.session_state.get('user_prefix','User: ')}{content}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='msg assistant'><div class='assistant-card'><div class='assistant-meta'>Assistant • Source-aware</div><div class='answer-content'>{content}</div></div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Input area
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    question = st.text_input("", placeholder='Ask a question about your uploaded documents...', key='question_input')
    submit = st.button("Send")

    if submit and question:
        st.session_state.messages.append({'role': 'user', 'content': question})
        if not uploaded_files:
            st.warning("Please upload at least one PDF before asking a question.")
        else:
            with st.spinner('Processing documents and building index...'):
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                saved_paths = []
                prog = st.progress(0)
                for i, f in enumerate(uploaded_files, start=1):
                    path = os.path.join(UPLOAD_DIR, f.name)
                    with open(path, 'wb') as out:
                        out.write(f.getbuffer())
                    saved_paths.append(path)
                    prog.progress(int(i / len(uploaded_files) * 100))

                process_documents(saved_paths)
                build_from_chunks_json()

            with st.spinner('Retrieving relevant chunks...'):
                chunks = retrieve_chunks(question)

            with st.spinner('Generating answer...'):
                answer = ask_llm(question, chunks)

            highlighted = answer
            for term in set(question.split()):
                if len(term) > 3:
                    highlighted = highlighted.replace(term, f"<strong style='background:rgba(37,99,235,0.12);padding:2px 4px;border-radius:4px'>{term}</strong>")

            st.session_state.messages.append({'role': 'assistant', 'content': highlighted})
            # Immediately render the assistant response in the main chat area so it's visible
            st.markdown(f"<div class='msg assistant'><div class='assistant-card'><div class='assistant-meta'>Assistant • Source-aware</div><div class='answer-content'>{highlighted}</div></div></div>", unsafe_allow_html=True)
            try:
                doc_names = [f.name for f in uploaded_files]
            except Exception:
                doc_names = [os.path.basename(p) for p in saved_paths] if 'saved_paths' in locals() else []
            st.session_state.history.append({'docs': doc_names, 'question': question, 'answer': highlighted})
            try:
                st.session_state['question_input'] = ''
            except Exception:
                pass

    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='side-card'>", unsafe_allow_html=True)
    st.markdown("<div class='side-title'>Previous Chats</div>", unsafe_allow_html=True)
    if st.session_state.history:
        for i, h in enumerate(reversed(st.session_state.history), start=1):
            with st.expander(f"Chat #{len(st.session_state.history)-i+1}: {h['question']}"):
                st.markdown(f"**Question:** {h['question']}")
                st.markdown(f"**Answer:** {h['answer']}")
                st.markdown("**Documents:**")
                for d in h.get('docs', []):
                    st.markdown(f"- {d}")
    else:
        st.markdown("<div style='color:var(--muted)'>No previous chats yet.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
