import os
from typing import Any

import requests
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = (5, 240)

st.set_page_config(
    page_title="Atlas · Knowledge Assistant",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A self-contained theme: no external assets or paid services are required.
st.markdown(
    """
    <style>
      :root {
        --ink: #18202f;
        --muted: #6b7485;
        --paper: #f7f8fc;
        --line: #e7eaf1;
        --violet: #6d5ce8;
        --violet-dark: #5545cb;
        --violet-soft: #f0edff;
        --mint: #1e9b78;
      }
      html, body, [class*="css"] { font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif; }
      .stApp { background: radial-gradient(ellipse at 70% 0%, #f0eeff 0%, #f7f8fc 44%, #f7f8fc 100%); color: var(--ink); }
      [data-testid="stHeader"] { background: rgba(247,248,252,.78); }
      .block-container { max-width: 1380px; padding-top: 1.55rem; padding-bottom: 2rem; }
      section[data-testid="stSidebar"] { background: #fff; border-right: 1px solid var(--line); }
      section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }
      section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .75rem; }
      h1, h2, h3 { font-family: 'Manrope', 'DM Sans', sans-serif !important; color: var(--ink); letter-spacing: -0.035em; }
      h1 { font-weight: 800 !important; }
      h2, h3 { font-weight: 750 !important; }
      p, label, .stMarkdown { color: var(--ink); }
      .brand-row { display: flex; align-items: center; gap: 10px; padding: 4px 0 18px; }
      .brand-mark { width: 38px; height: 38px; border-radius: 13px; display: flex; align-items: center; justify-content: center; color: #fff; font-size: 22px; font-weight: 700; background: linear-gradient(135deg, #8b7bff 0%, #5848cf 100%); box-shadow: 0 8px 18px rgba(109,92,232,.24); }
      .brand-name { font-family: 'Manrope', sans-serif; color: #20263a; font-weight: 800; font-size: 18px; letter-spacing: -.045em; line-height: 1.1; }
      .brand-sub { color: #858c9b; font-size: 10px; text-transform: uppercase; letter-spacing: .14em; margin-top: 4px; }
      .side-label { color: #7c8393; font-weight: 700; font-size: 10px; letter-spacing: .14em; text-transform: uppercase; margin: .55rem 0 .15rem; }
      .upload-note { border: 1px dashed #cbc6fa; border-radius: 14px; background: linear-gradient(135deg, #faf9ff, #f5f3ff); padding: 13px 14px; color: #666287; font-size: 12px; line-height: 1.55; margin: 2px 0 4px; }
      .topline { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 10px; flex-wrap: wrap; }
      .eyebrow { color: var(--violet-dark); font-weight: 800; font-size: 10px; letter-spacing: .16em; text-transform: uppercase; }
      .local-pill { display: inline-flex; align-items: center; gap: 7px; border: 1px solid #cdeee2; background: #effbf6; color: #18785f; border-radius: 999px; padding: 6px 11px; font-size: 11px; font-weight: 700; }
      .local-dot { width: 7px; height: 7px; border-radius: 99px; background: #26a77d; box-shadow: 0 0 0 3px rgba(38,167,125,.12); }
      .hero-title { font-family: 'Manrope', sans-serif; color: #1b2234; font-size: clamp(2.05rem, 3.3vw, 3.15rem); font-weight: 800; letter-spacing: -.065em; line-height: 1.08; margin: 8px 0 12px; }
      .hero-title span { color: var(--violet); }
      .hero-copy { color: #737b8c; font-size: 15px; line-height: 1.65; max-width: 700px; margin-bottom: 24px; }
      .metric-card { background: rgba(255,255,255,.88); border: 1px solid var(--line); border-radius: 16px; padding: 15px 17px; min-height: 94px; box-shadow: 0 5px 20px rgba(26,35,65,.025); }
      .metric-label { color: #838a99; font-size: 11px; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; margin-bottom: 7px; }
      .metric-value { font-family: 'Manrope', sans-serif; color: #222a3c; font-size: 25px; font-weight: 800; letter-spacing: -.04em; line-height: 1.1; }
      .metric-foot { color: #858b99; font-size: 11px; margin-top: 5px; }
      .section-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin: 26px 0 10px; }
      .section-title { font-family: 'Manrope', sans-serif; font-weight: 800; color: #22293a; font-size: 18px; letter-spacing: -.04em; }
      .section-kicker { color: #858b99; font-size: 12px; }
      .empty-card { border: 1px solid #e6e8f1; background: rgba(255,255,255,.78); border-radius: 22px; padding: 27px 25px 22px; text-align: center; margin: 10px 0 18px; }
      .empty-icon { margin: 0 auto 12px; width: 54px; height: 54px; display: flex; align-items: center; justify-content: center; border-radius: 17px; background: var(--violet-soft); color: var(--violet-dark); font-size: 25px; }
      .empty-title { color: #22293a; font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 20px; letter-spacing: -.04em; margin-bottom: 7px; }
      .empty-copy { color: #798193; max-width: 550px; margin: 0 auto; line-height: 1.6; font-size: 13px; }
      .hint { background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 13px 15px; color: #515b6e; font-size: 12px; line-height: 1.5; }
      [data-testid="stChatMessage"] { border: 1px solid #e8eaf2; border-radius: 18px; padding: 14px 16px; background: rgba(255,255,255,.86); box-shadow: 0 5px 18px rgba(26,35,65,.025); margin-bottom: 10px; }
      [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { line-height: 1.68; }
      [data-testid="stChatInput"] { border-color: #dcdfea !important; border-radius: 16px !important; box-shadow: 0 8px 28px rgba(29,36,67,.07); background: #fff !important; }
      [data-testid="stChatInput"] textarea { font-size: 14px !important; }
      .stButton > button { border-radius: 11px; font-weight: 700; border: 1px solid #e2e5ef; transition: all .15s ease; }
      .stButton > button[kind="primary"] { background: var(--violet); border-color: var(--violet); color: white; box-shadow: 0 5px 13px rgba(109,92,232,.18); }
      .stButton > button[kind="primary"]:hover { background: var(--violet-dark); border-color: var(--violet-dark); }
      .stButton > button:hover { border-color: #c6c0fb; color: var(--violet-dark); }
      [data-testid="stFileUploader"] { background: #fcfbff; border: 1px dashed #cbc6fa; border-radius: 13px; padding: 6px; }
      [data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 12px; background: rgba(255,255,255,.85); }
      [data-testid="stExpander"] summary { font-size: 12px; font-weight: 700; }
      hr { border-color: #eceef4; }
      .source-meta { color: #80889a; font-size: 11px; }
      .footer-note { color: #9aa0ad; font-size: 11px; text-align: center; margin: 26px 0 5px; }
      #MainMenu, footer { visibility: hidden; }
      @media (max-width: 700px) {
        .block-container { padding-top: 1rem; }
        .hero-title { font-size: 2.15rem; }
        .hero-copy { font-size: 13px; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_error_message(response: requests.Response) -> str:
    """Read FastAPI's usual detail field without dumping a large HTML error page."""
    try:
        payload = response.json()
        detail = payload.get("detail", payload)
        return str(detail)
    except (ValueError, AttributeError):
        return f"Request failed with HTTP {response.status_code}."


def get_documents() -> list[dict[str, Any]]:
    try:
        response = requests.get(f"{API_URL}/documents", timeout=8)
        response.raise_for_status()
        payload = response.json()
        return payload.get("documents", []) if isinstance(payload, dict) else []
    except (requests.RequestException, ValueError):
        return []


def get_api_status() -> tuple[bool, str]:
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        response.raise_for_status()
        data = response.json()
        return data.get("status") == "ok", data.get("database", "unknown")
    except (requests.RequestException, ValueError):
        return False, "unavailable"


def render_citations(citations: list[dict[str, Any]], message_index: int) -> None:
    if not citations:
        st.caption("No supporting passages were returned for this answer.")
        return

    st.markdown("<div class='side-label'>Sources & evidence</div>", unsafe_allow_html=True)
    for citation in citations:
        citation_id = citation.get("citation_id", "?")
        source = citation.get("source", "Unknown document")
        page = citation.get("page", "?")
        snippet = citation.get("snippet", "No snippet available.")
        score = citation.get("retrieval_score")
        label = f"[{citation_id}]  {source}  ·  Page {page}"
        with st.expander(label, expanded=False):
            st.markdown(f"**{source}** · Page {page}")
            st.write(snippet)
            if score is not None:
                st.caption(f"Retrieval ranking score: {float(score):.4f}")


# Session state persists the conversation while Streamlit reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "last_latency_ms" not in st.session_state:
    st.session_state.last_latency_ms = None
if "upload_notice" not in st.session_state:
    st.session_state.upload_notice = None

# Sidebar: brand, ingestion, and document library.
with st.sidebar:
    st.markdown(
        """
        <div class="brand-row">
          <div class="brand-mark">✳</div>
          <div><div class="brand-name">Atlas</div><div class="brand-sub">Knowledge workspace</div></div>
        </div>
        <div class="side-label">Your knowledge base</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='upload-note'>Add a text-based PDF. Atlas will index its content and use it as evidence when answering your questions.</div>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        accept_multiple_files=False,
        help="Text-based PDFs up to 20 MB are supported in this version.",
        label_visibility="collapsed",
    )
    if st.button(
        "＋  Upload and index PDF",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    ):
        try:
            with st.spinner("Reading PDF and building local embeddings…"):
                response = requests.post(
                    f"{API_URL}/documents/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    timeout=REQUEST_TIMEOUT,
                )
            if response.ok:
                result = response.json()
                st.session_state.upload_notice = {
                    "ok": True,
                    "text": f"{result.get('filename', uploaded_file.name)} indexed · {result.get('chunks_added', 0)} chunks",
                }
            else:
                st.session_state.upload_notice = {"ok": False, "text": api_error_message(response)}
        except requests.RequestException as exc:
            st.session_state.upload_notice = {
                "ok": False,
                "text": f"Could not reach the API at {API_URL}. Details: {exc}",
            }
        st.rerun()

    if st.session_state.upload_notice:
        notice = st.session_state.upload_notice
        if notice["ok"]:
            st.success(notice["text"])
        else:
            st.error(notice["text"])

    st.divider()
    st.markdown("<div class='side-label'>Indexed documents</div>", unsafe_allow_html=True)
    documents = get_documents()
    total_chunks = sum(int(doc.get("chunks", 0)) for doc in documents)
    if documents:
        st.caption(f"{len(documents)} document{'s' if len(documents) != 1 else ''} · {total_chunks} indexed chunks")
        for document in documents:
            filename = document.get("filename", "Untitled PDF")
            chunks = document.get("chunks", 0)
            first_page = document.get("first_page")
            last_page = document.get("last_page")
            with st.expander(f"📄  {filename}", expanded=False):
                st.markdown(f"**{chunks} chunks**")
                if first_page is not None and last_page is not None:
                    st.caption(f"Pages represented: {first_page}–{last_page}")
    else:
        st.caption("Your library is empty. Upload a PDF to get started.")

    st.divider()
    api_ok, db_status = get_api_status()
    if api_ok:
        st.markdown(
            f"<div class='hint'>🟢 <b>API connected</b><br><span style='color:#7d8494'>Database: {db_status}</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='hint'>🔴 <b>API unavailable</b><br><span style='color:#7d8494'>Start the backend with Docker Compose.</span></div>",
            unsafe_allow_html=True,
        )
    st.caption("Local-first · FastAPI · pgvector · Ollama")

# Main page header.
api_ok, _db_status = get_api_status()
st.markdown(
    f"""
    <div class="topline">
      <div class="eyebrow">DOCUMENT INTELLIGENCE, WITHOUT THE NOISE</div>
      <div class="local-pill"><span class="local-dot"></span> LOCAL AI MODE · NO OPENAI CALLS</div>
    </div>
    <div class="hero-title">Make your documents<br><span>answer back.</span></div>
    <div class="hero-copy">Ask natural questions across your PDFs. Get grounded answers, inspect the evidence, and keep your workflow on your own machine.</div>
    """,
    unsafe_allow_html=True,
)

# Quick workspace stats.
number_documents = len(documents)
number_chunks = total_chunks
last_latency = st.session_state.last_latency_ms
latency_text = f"{last_latency / 1000:.1f}s" if isinstance(last_latency, (int, float)) else "—"
metric_cols = st.columns(3)
metrics = [
    ("DOCUMENTS INDEXED", str(number_documents), "PDFs in your library"),
    ("SEARCHABLE CHUNKS", str(number_chunks), "Passages ready for retrieval"),
    ("LAST RESPONSE", latency_text, "Local model response time"),
]
for column, (label, value, foot) in zip(metric_cols, metrics):
    with column:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-foot'>{foot}</div></div>",
            unsafe_allow_html=True,
        )

st.markdown(
    "<div class='section-head'><div class='section-title'>Ask your knowledge base</div><div class='section-kicker'>Answers include source evidence when available</div></div>",
    unsafe_allow_html=True,
)

# Conversation history.
if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-card">
          <div class="empty-icon">✧</div>
          <div class="empty-title">A better way to explore your PDFs</div>
          <div class="empty-copy">Start with a question below, or try one of these prompts. Atlas searches your indexed passages first, then asks the local model to answer from the evidence.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    prompt_cols = st.columns(3)
    example_questions = [
        "Summarize the main ideas in my documents",
        "What skills and projects are mentioned?",
        "What information is missing from the documents?",
    ]
    for column, prompt in zip(prompt_cols, example_questions):
        with column:
            if st.button(prompt, key=f"suggest_{prompt}", use_container_width=True):
                st.session_state.pending_question = prompt
                st.rerun()
else:
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"], avatar="✳" if message["role"] == "assistant" else "🙂"):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                meta = message.get("meta", {})
                if meta:
                    st.caption(
                        f"{meta.get('latency_ms', 0) / 1000:.1f}s · {meta.get('retrieval_count', 0)} cited passage(s) retrieved"
                    )
                render_citations(message.get("citations", []), index)

question_from_input = st.chat_input("Ask a question about your indexed PDFs…")
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
else:
    question = question_from_input

if question:
    cleaned_question = question.strip()
    if len(cleaned_question) < 3:
        st.warning("Enter a question with at least three characters.")
    else:
        st.session_state.messages.append({"role": "user", "content": cleaned_question})
        try:
            with st.spinner("Searching your documents and composing an evidence-based answer…"):
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": cleaned_question},
                    timeout=REQUEST_TIMEOUT,
                )
            if response.ok:
                result = response.json()
                latency_ms = int(result.get("latency_ms", 0))
                citations = result.get("citations", [])
                answer = result.get("answer", "The API returned no answer text.")
                st.session_state.last_latency_ms = latency_ms
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "meta": {
                            "latency_ms": latency_ms,
                            "retrieval_count": result.get("retrieval_count", len(citations)),
                        },
                    }
                )
            else:
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"I couldn't complete that request. {api_error_message(response)}", "citations": [], "meta": {}}
                )
        except requests.RequestException as exc:
            st.session_state.messages.append(
                {"role": "assistant", "content": f"I couldn't reach the API at `{API_URL}`. Start the backend and try again.\n\nDetails: {exc}", "citations": [], "meta": {}}
            )
        st.rerun()

footer_status = "Connected to local RAG backend" if api_ok else "Backend connection unavailable"
st.markdown(
    f"<div class='footer-note'>Atlas · {footer_status} · Your documents are searched through your configured local backend.</div>",
    unsafe_allow_html=True,
)
