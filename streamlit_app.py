"""Streamlit UI for the Multimodal AI Research Assistant FastAPI service."""
from __future__ import annotations

import os
from pathlib import Path

import requests
import streamlit as st

st.set_page_config(page_title="Multimodal Research Assistant", page_icon="📚", layout="wide")

API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.title("📚 Multimodal AI Research Assistant")
st.caption("Ask questions about the indexed research paper. Answers are grounded in retrieved text, tables, and figures.")

with st.sidebar:
    st.subheader("Connection")
    st.code(API_URL, language=None)
    if st.button("Check API health", use_container_width=True):
        try:
            response = requests.get(f"{API_URL}/health", timeout=10)
            response.raise_for_status()
            health = response.json()
            if health.get("status") == "ok":
                st.success("API is running")
            else:
                st.warning("API is not ready yet")
            st.json(health)
        except requests.RequestException as exc:
            st.error(f"Could not reach API: {exc}")
    st.markdown("**Run the API first:**\n```bash\nuv run uvicorn api:app --reload\n```")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("result"):
            result = message["result"]
            with st.expander("Retrieved text evidence"):
                for index, source in enumerate(result.get("text_sources", []), start=1):
                    st.markdown(f"**{index}. Physical PDF page {source.get('page', 'unknown')}** · score: {source.get('score', 'n/a')}")
                    st.write(source.get("text", ""))
            with st.expander("Retrieved tables"):
                for table in result.get("tables", []):
                    st.markdown(f"**Physical PDF page {table.get('page', 'unknown')}** · score: {table.get('score', 'n/a')}")
                    st.code(table.get("text", ""), language=None)
            with st.expander("Retrieved figures and descriptions"):
                desc_by_path = {item.get("image_path"): item for item in result.get("figure_descriptions", [])}
                for figure in result.get("figures", []):
                    st.markdown(f"**Physical PDF page {figure.get('page', 'unknown')}** · score: {figure.get('score', 'n/a')}")
                    image_path = Path(figure.get("image_path", ""))
                    if figure.get("exists") and image_path.is_file():
                        st.image(str(image_path), use_container_width=True)
                    description = desc_by_path.get(figure.get("image_path"), {}).get("description")
                    if description:
                        st.write(description)

prompt = st.chat_input("Ask a question about the research paper...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving text, tables, and figures and preparing a grounded answer..."):
            try:
                response = requests.post(
                    f"{API_URL}/api/query",
                    json={"query": prompt},
                    timeout=240,
                )
                if not response.ok:
                    try:
                        detail = response.json().get("detail", response.text)
                    except ValueError:
                        detail = response.text
                    st.error(f"API error ({response.status_code}): {detail}")
                    st.stop()
                result = response.json()
                answer = result.get("answer", "No answer returned.")
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer, "result": result})
                st.rerun()
            except requests.Timeout:
                st.error("The request timed out. First model calls may be slow; check the API terminal and try again.")
            except requests.RequestException as exc:
                st.error(f"Could not reach the API. Start it with `uv run uvicorn api:app --reload`. Details: {exc}")

if not st.session_state.messages:
    st.info("Start with a question such as: **What was Pakistan's average annual population growth rate in 2017–2023?**")
