"""Streamlit UI for uploading a PDF and asking grounded research questions."""
from __future__ import annotations

import base64
import hashlib
import os
import time
import uuid
from io import BytesIO
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

st.set_page_config(page_title="Multimodal AI Research Assistant", page_icon="📚", layout="wide")


def _secret(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default


API_URL = _secret("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_TOKEN = _secret("RAG_API_TOKEN", "")

st.title("📚 Multimodal AI Research Assistant")
st.caption("Upload a research paper and ask questions grounded in its text, tables, and figures.")

with st.sidebar:
    st.subheader("Backend connection")
    st.code(API_URL, language=None)
    if st.button("Check API health", use_container_width=True):
        try:
            response = requests.get(f"{API_URL}/health", timeout=20)
            response.raise_for_status()
            health = response.json()
            st.success("API is reachable")
            st.json(health)
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
    st.caption("The first question after inactivity may take longer while Azure starts the backend and loads AI models.")

uploaded_file = st.file_uploader(
    "Upload a research paper (PDF, up to 20 MB and 60 pages)",
    type=["pdf"],
    help="Scanned PDFs without selectable text are not supported by this demo yet.",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "pdf_sha" not in st.session_state:
    st.session_state.pdf_sha = ""
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = b""
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = "research-paper.pdf"

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    current_sha = hashlib.sha256(pdf_bytes).hexdigest()
    if current_sha != st.session_state.pdf_sha:
        st.session_state.pdf_sha = current_sha
        st.session_state.pdf_bytes = pdf_bytes
        st.session_state.pdf_name = uploaded_file.name
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.messages = []
    else:
        st.session_state.pdf_bytes = pdf_bytes
        st.session_state.pdf_name = uploaded_file.name

if st.session_state.pdf_bytes:
    st.success(f"Paper selected: **{st.session_state.pdf_name}** · {len(st.session_state.pdf_bytes) / (1024 * 1024):.2f} MB")
    st.info("Ask a question below. Indexing happens automatically on the first question.")
else:
    st.info("Upload a PDF to begin. The assistant will build an index for that paper automatically.")


def _render_evidence(result: dict) -> None:
    with st.expander("Retrieved text evidence"):
        for i, source in enumerate(result.get("text_sources", []), start=1):
            st.markdown(f"**{i}. Physical PDF page {source.get('page', 'unknown')}** · score: {source.get('score', 'n/a')}")
            st.write(source.get("text", ""))
        if not result.get("text_sources"):
            st.write("No relevant text chunks were returned.")
    with st.expander("Retrieved tables"):
        for table in result.get("tables", []):
            st.markdown(f"**Physical PDF page {table.get('page', 'unknown')}** · score: {table.get('score', 'n/a')}")
            st.code(table.get("text", ""), language=None)
        if not result.get("tables"):
            st.write("No table was extracted or retrieved for this question.")
    with st.expander("Retrieved figures and descriptions"):
        descriptions = result.get("figure_descriptions", [])
        descriptions_by_page = {str(item.get("page")): item.get("description", "") for item in descriptions}
        for figure in result.get("figures", []):
            page = str(figure.get("page", "unknown"))
            st.markdown(f"**Physical PDF page {page}** · score: {figure.get('score', 'n/a')}")
            encoded = figure.get("image_base64")
            if encoded:
                try:
                    st.image(BytesIO(base64.b64decode(encoded)), use_container_width=True)
                except Exception:
                    st.caption("Could not display this retrieved figure.")
            description = descriptions_by_page.get(page)
            if description:
                st.write(description)
        if not result.get("figures"):
            st.write("No relevant embedded figure was retrieved for this question.")
    if result.get("index_stats"):
        with st.expander("Document indexing details"):
            st.json(result["index_stats"])


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("result"):
            _render_evidence(message["result"])

prompt = st.chat_input("Ask a question about the uploaded paper...", disabled=not bool(st.session_state.pdf_bytes))
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Reading the paper, retrieving evidence, and preparing an answer. The first request can take a few minutes..."):
            try:
                headers = {"X-RAG-Token": API_TOKEN} if API_TOKEN else {}
                files = {"file": (st.session_state.pdf_name, st.session_state.pdf_bytes, "application/pdf")}
                data = {"query": prompt, "session_id": st.session_state.session_id}
                request_started = time.perf_counter()
                response = requests.post(
                    f"{API_URL}/api/ask",
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=600,
                )
                request_elapsed = time.perf_counter() - request_started
                if not response.ok:
                    try:
                        detail = response.json().get("detail", response.text)
                    except ValueError:
                        detail = response.text
                    st.error(f"API error ({response.status_code}): {detail}")
                else:
                    result = response.json()
                    answer = result.get("answer", "No answer was returned.")
                    st.markdown(answer)
                    st.caption(
                        f"Request completed in {request_elapsed:.1f}s, including backend startup "
                        "(if needed), model loading, and document indexing."
                    )
                    st.session_state.messages.append({"role": "assistant", "content": answer, "result": result})
                    _render_evidence(result)
            except requests.Timeout:
                st.error("The request timed out. The backend may still be downloading/loading models. Check Azure logs and retry.")
            except requests.RequestException as exc:
                st.error(f"Could not reach the API: {exc}")
