"""
Intelligent Document Q&A Assistant — Member 3 starter skeleton
Run with: streamlit run app.py

This is a STANDALONE demo. The retrieve_chunks() function below is a
placeholder — once Member 2 builds the real embedding + FAISS/Chroma
retrieval pipeline, swap it in here without touching anything else.
"""

import streamlit as st
import ollama

st.set_page_config(page_title="Document Q&A Assistant", page_icon="📄")
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
    return [
        {
            "text": "The default OSPF hello interval is 10 seconds on "
                    "broadcast and point-to-point networks, and 30 seconds "
                    "on non-broadcast multi-access (NBMA) networks.",
            "source_file": "Networking_User_Guide.pdf",
            "page": 42,
        }
    ]

# ---------------------------------------------------------------------
# 3. LLM call — answers ONLY from retrieved chunks, with citations
# ---------------------------------------------------------------------
def ask_llm(question: str, chunks: list) -> str:
    context = "\n\n".join(
        f"[Source: {c['source_file']}, page {c['page']}]\n{c['text']}"
        for c in chunks
    )
    prompt = f"""You are a documentation assistant. Answer the question
using ONLY the context below. Be specific — include exact numbers, names,
or values from the context rather than vague restatements. If the answer
isn't in the context, say "I couldn't find this in the uploaded documents."
Always cite the source file and page number at the end of your answer.

Context:
{context}

Question: {question}
Answer:"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
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
                            st.markdown(f"**{c['source_file']}**, page {c['page']}")
                            st.caption(c["text"])
                except Exception as e:
                    answer = (
                        "Something went wrong while generating the answer. "
                        "Make sure the Ollama app is running in the background."
                    )
                    st.error(answer)
                    st.caption(f"Technical details: {e}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
