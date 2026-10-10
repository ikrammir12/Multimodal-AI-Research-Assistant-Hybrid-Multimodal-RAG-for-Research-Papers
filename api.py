"""Upload-based multimodal RAG API designed for a low-traffic, scale-to-zero demo.

The PDF is sent with each question. A short-lived in-memory cache avoids re-indexing
while the same Container Apps replica is alive. If the app scales to zero, the next
request includes the PDF again and the index is rebuilt automatically; no persistent
ChromaDB volume is required for this demo setup.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import io
import os
import re
import time
import threading
from dataclasses import dataclass, field
from typing import Any

import fitz
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from sentence_transformers import SentenceTransformer
import psutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

EMBEDDING_MODEL_NAME = os.getenv("BGE_MODEL_NAME", "BAAI/bge-m3")
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "sentence-transformers/clip-ViT-B-32")
MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 60
MAX_CHUNKS = 600
MAX_FIGURES = 25
MAX_CACHE_SESSIONS = 2
CACHE_TTL_SECONDS = 30 * 60
TEXT_TOP_K = 5
TABLE_TOP_K = 3
FIGURE_TOP_K = 3
MAX_IMAGES_FOR_VLM = 2

app = FastAPI(
    title="Multimodal AI Research Assistant API",
    version="2.0.0",
    description=(
        "Upload a research PDF and ask questions grounded in retrieved text, tables, and figures. "
        "Optimized for low traffic with an in-memory, rebuildable document index."
    ),
)

# Models are loaded lazily on first use and reused for later requests handled by
# this process. Locks prevent simultaneous first requests from loading duplicates.
_embedding_model: SentenceTransformer | None = None
_clip_model: SentenceTransformer | None = None
_answer_llm: Any = None
_vision_llm: Any = None
_embedding_lock = threading.Lock()
_clip_lock = threading.Lock()
_openai_lock = threading.Lock()
_embedding_model_load_seconds: float | None = None
_clip_model_load_seconds: float | None = None
_process_started_at = time.monotonic()


@app.middleware("http")
async def log_request_metrics(request, call_next):
    """Log endpoint duration and RSS without logging questions, files, or secrets."""
    started = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        duration = round(time.perf_counter() - started, 2)
        status = response.status_code if response is not None else 500
        print(
            f"HTTP {request.method} {request.url.path} status={status} "
            f"duration_seconds={duration} process_rss_mb={_memory_usage_mb()}"
        )


def _memory_usage_mb() -> float | None:
    try:
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").lower())


def _check_api_token(x_rag_token: str | None) -> None:
    """Protect the public API from casual direct calls and unexpected API usage."""
    expected = os.getenv("RAG_API_TOKEN", "").strip()
    production = os.getenv("APP_ENV", "").lower() == "production"
    if production and not expected:
        raise HTTPException(status_code=503, detail="RAG_API_TOKEN is not configured on the backend.")
    if expected and (not x_rag_token or not hmac.compare_digest(x_rag_token, expected)):
        raise HTTPException(status_code=401, detail="Missing or invalid API access token.")


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model, _embedding_model_load_seconds
    if _embedding_model is None:
        with _embedding_lock:
            if _embedding_model is None:
                started = time.perf_counter()
                print(
                    f"Loading text embedding model: {EMBEDDING_MODEL_NAME}. "
                    "If it is not cached, weights will be downloaded from Hugging Face."
                )
                _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
                _embedding_model_load_seconds = round(time.perf_counter() - started, 2)
                print(
                    f"Text embedding model ready in {_embedding_model_load_seconds}s; "
                    f"process_rss_mb={_memory_usage_mb()}"
                )
    return _embedding_model


def _get_clip_model() -> SentenceTransformer:
    global _clip_model, _clip_model_load_seconds
    if _clip_model is None:
        with _clip_lock:
            if _clip_model is None:
                started = time.perf_counter()
                print(
                    f"Loading figure embedding model: {CLIP_MODEL_NAME}. "
                    "If it is not cached, weights will be downloaded from Hugging Face."
                )
                _clip_model = SentenceTransformer(CLIP_MODEL_NAME, device="cpu")
                _clip_model_load_seconds = round(time.perf_counter() - started, 2)
                print(
                    f"Figure embedding model ready in {_clip_model_load_seconds}s; "
                    f"process_rss_mb={_memory_usage_mb()}"
                )
    return _clip_model


def _get_openai_models() -> tuple[Any, Any]:
    global _answer_llm, _vision_llm
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured on the backend.",
        )
    if _answer_llm is None or _vision_llm is None:
        with _openai_lock:
            if _answer_llm is None or _vision_llm is None:
                try:
                    from langchain_openai import ChatOpenAI

                    _answer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=api_key)
                    _vision_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
                except Exception as exc:
                    raise HTTPException(status_code=503, detail=f"Could not initialize OpenAI clients: {exc}") from exc
    return _answer_llm, _vision_llm


def _make_chunks(page_text: str, page_num: int, size: int = 900, overlap: int = 120) -> list[dict[str, Any]]:
    text = (page_text or "").strip()
    if not text:
        return []
    chunks: list[dict[str, Any]] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Prefer a nearby line/word boundary so a chunk is not cut mid-word.
            split_at = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
            if split_at > start + int(size * 0.6):
                end = split_at
        piece = text[start:end].strip()
        if piece:
            chunks.append({"kind": "text", "page": page_num, "text": piece})
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _extract_tables(page: fitz.Page, page_num: int) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    finder = getattr(page, "find_tables", None)
    if finder is None:
        return found
    try:
        table_finder = finder()
        tables = getattr(table_finder, "tables", []) or []
        for number, table in enumerate(tables, start=1):
            rows = table.extract() or []
            lines = []
            for row in rows:
                cells = [str(cell).strip() if cell is not None else "" for cell in row]
                if any(cells):
                    lines.append(" | ".join(cells))
            table_text = "\n".join(lines).strip()
            if len(table_text) >= 20:
                found.append({
                    "kind": "table",
                    "page": page_num,
                    "table_number": number,
                    "text": table_text[:10000],
                })
    except Exception as exc:
        # Table extraction is PDF-layout dependent; text retrieval should still work.
        print(f"Table extraction skipped on page {page_num}: {exc}")
    return found


def _extract_figures(pdf: fitz.Document) -> list[dict[str, Any]]:
    figures: list[dict[str, Any]] = []
    seen_on_page: set[tuple[int, int]] = set()
    for page_index, page in enumerate(pdf, start=1):
        try:
            image_entries = page.get_images(full=True)
        except Exception:
            continue
        for entry in image_entries:
            if len(figures) >= MAX_FIGURES:
                return figures
            xref = int(entry[0])
            key = (page_index, xref)
            if key in seen_on_page:
                continue
            seen_on_page.add(key)
            try:
                if not page.get_image_rects(xref):
                    continue
                extracted = pdf.extract_image(xref)
                raw = extracted.get("image", b"")
                if not raw:
                    continue
                with Image.open(io.BytesIO(raw)) as source:
                    image = source.convert("RGB")
                if image.width < 120 or image.height < 120:
                    continue
                image.thumbnail((1600, 1600))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=88, optimize=True)
                jpeg_bytes = buffer.getvalue()
                if len(jpeg_bytes) > 2 * 1024 * 1024:
                    continue
                figures.append({
                    "page": page_index,
                    "bytes": jpeg_bytes,
                    "mime_type": "image/jpeg",
                    "width": image.width,
                    "height": image.height,
                })
            except (UnidentifiedImageError, ValueError, RuntimeError, KeyError) as exc:
                print(f"Skipping an embedded image on page {page_index}: {exc}")
            except Exception as exc:
                print(f"Skipping an embedded image on page {page_index}: {exc}")
    return figures


@dataclass
class DocumentIndex:
    pdf_sha256: str
    filename: str
    documents: list[dict[str, Any]]
    embeddings: np.ndarray
    bm25: Any
    figures: list[dict[str, Any]] = field(default_factory=list)
    figure_embeddings: np.ndarray | None = None
    last_used: float = field(default_factory=time.time)


_indexes: dict[str, DocumentIndex] = {}


def _evict_old_indexes() -> None:
    now = time.time()
    expired = [sid for sid, index in _indexes.items() if now - index.last_used > CACHE_TTL_SECONDS]
    for sid in expired:
        _indexes.pop(sid, None)
    while len(_indexes) >= MAX_CACHE_SESSIONS:
        oldest = min(_indexes, key=lambda sid: _indexes[sid].last_used)
        _indexes.pop(oldest, None)


def _build_index(pdf_bytes: bytes, filename: str, pdf_sha: str) -> DocumentIndex:
    try:
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not open this PDF: {exc}") from exc

    indexing_started = time.perf_counter()
    with pdf:
        if pdf.page_count == 0:
            raise HTTPException(status_code=422, detail="The PDF has no pages.")
        if pdf.page_count > MAX_PDF_PAGES:
            raise HTTPException(
                status_code=413,
                detail=f"This demo supports up to {MAX_PDF_PAGES} pages per PDF. Your file has {pdf.page_count} pages.",
            )
        documents: list[dict[str, Any]] = []
        for page_index, page in enumerate(pdf, start=1):
            page_text = page.get_text("text") or ""
            documents.extend(_make_chunks(page_text, page_index))
            documents.extend(_extract_tables(page, page_index))
        figures = _extract_figures(pdf)

    text_doc_count = sum(doc["kind"] == "text" for doc in documents)
    if text_doc_count == 0:
        raise HTTPException(
            status_code=422,
            detail="No selectable text was found. This demo does not currently run OCR on scanned PDFs.",
        )
    if len(documents) > MAX_CHUNKS:
        documents = documents[:MAX_CHUNKS]

    print(f"Indexing {filename}: {text_doc_count} text chunks, "
          f"{sum(doc['kind'] == 'table' for doc in documents)} tables, {len(figures)} figures.")
    embedding_started = time.perf_counter()
    model = _get_embedding_model()
    vectors = model.encode(
        [doc["text"] for doc in documents],
        batch_size=8,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embedding_seconds = round(time.perf_counter() - embedding_started, 2)
    print(f"Document text/table embedding finished in {embedding_seconds}s; process_rss_mb={_memory_usage_mb()}")

    bm25 = None
    try:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi([_tokenize(doc["text"]) for doc in documents])
    except Exception as exc:
        print(f"BM25 is unavailable; dense retrieval will be used: {exc}")

    index = DocumentIndex(
        pdf_sha256=pdf_sha,
        filename=filename,
        documents=documents,
        embeddings=np.asarray(vectors, dtype=np.float32),
        bm25=bm25,
        figures=figures,
    )
    if figures:
        try:
            figure_embedding_started = time.perf_counter()
            clip = _get_clip_model()
            pil_images = []
            for figure in figures:
                with Image.open(io.BytesIO(figure["bytes"])) as image:
                    pil_images.append(image.convert("RGB"))
            image_vectors = clip.encode(
                pil_images,
                batch_size=4,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            index.figure_embeddings = np.asarray(image_vectors, dtype=np.float32)
            print(
                f"Figure embedding finished in {round(time.perf_counter() - figure_embedding_started, 2)}s; "
                f"process_rss_mb={_memory_usage_mb()}"
            )
        except Exception as exc:
            print(f"Figure embeddings unavailable for this PDF; continuing with text and tables: {exc}")
            index.figure_embeddings = None
    print(
        f"PDF indexing total: {round(time.perf_counter() - indexing_started, 2)}s; "
        f"process_rss_mb={_memory_usage_mb()}"
    )
    return index


def _get_index(session_id: str, filename: str, pdf_bytes: bytes) -> DocumentIndex:
    pdf_sha = hashlib.sha256(pdf_bytes).hexdigest()
    cached = _indexes.get(session_id)
    if cached and cached.pdf_sha256 == pdf_sha:
        cached.last_used = time.time()
        print(f"Document index cache hit for session {session_id[:8]}.")
        return cached
    _evict_old_indexes()
    index = _build_index(pdf_bytes, filename, pdf_sha)
    index.last_used = time.time()
    _indexes[session_id] = index
    return index


def _embed_query(text: str) -> np.ndarray:
    return _get_embedding_model().encode(
        [text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )[0].astype(np.float32)


def _hybrid_search(index: DocumentIndex, query: str, limit: int = TEXT_TOP_K) -> list[dict[str, Any]]:
    if not index.documents:
        return []
    query_vector = _embed_query(query)
    dense_scores = index.embeddings @ query_vector
    dense_ids = np.argsort(dense_scores)[::-1][:min(10, len(index.documents))].tolist()

    sparse_ids: list[int] = []
    if index.bm25 is not None:
        try:
            sparse_scores = np.asarray(index.bm25.get_scores(_tokenize(query)))
            sparse_ids = np.argsort(sparse_scores)[::-1][:min(10, len(index.documents))].tolist()
        except Exception as exc:
            print(f"BM25 search failed; using dense search: {exc}")

    rrf: dict[int, float] = {}
    for ranked_ids in (dense_ids, sparse_ids):
        for rank, item_id in enumerate(ranked_ids, start=1):
            rrf[item_id] = rrf.get(item_id, 0.0) + 1.0 / (60 + rank)
    ranked = sorted(rrf, key=rrf.get, reverse=True)
    return [
        {
            **index.documents[item_id],
            "score": float(dense_scores[item_id]),
            "rrf_score": float(rrf[item_id]),
        }
        for item_id in ranked if item_id < len(index.documents)
    ][:limit]


def _text_search(index: DocumentIndex, query: str, limit: int = TEXT_TOP_K) -> list[dict[str, Any]]:
    hybrid = _hybrid_search(index, query, limit=min(len(index.documents), max(limit * 4, 20)))
    results = [item for item in hybrid if item.get("kind") == "text"][:limit]
    if len(results) < limit:
        query_vector = _embed_query(query)
        scores = index.embeddings @ query_vector
        text_ids = [i for i, doc in enumerate(index.documents) if doc.get("kind") == "text"]
        existing_pages_and_text = {(item.get("page"), item.get("text")) for item in results}
        for item_id in sorted(text_ids, key=lambda i: scores[i], reverse=True):
            doc = index.documents[item_id]
            if (doc.get("page"), doc.get("text")) in existing_pages_and_text:
                continue
            results.append({**doc, "score": float(scores[item_id])})
            if len(results) >= limit:
                break
    return results


def _table_search(index: DocumentIndex, query: str, limit: int = TABLE_TOP_K) -> list[dict[str, Any]]:
    table_ids = [i for i, doc in enumerate(index.documents) if doc.get("kind") == "table"]
    if not table_ids:
        return []
    query_vector = _embed_query(query)
    scores = index.embeddings[table_ids] @ query_vector
    ranked = np.argsort(scores)[::-1][:min(limit, len(table_ids))]
    output = []
    for rank_index in ranked:
        doc_id = table_ids[int(rank_index)]
        doc = index.documents[doc_id]
        output.append({
            "page": doc.get("page"),
            "table_number": doc.get("table_number"),
            "text": doc.get("text", ""),
            "score": float(scores[int(rank_index)]),
        })
    return output


def _figure_search(index: DocumentIndex, query: str, limit: int = FIGURE_TOP_K) -> list[dict[str, Any]]:
    if not index.figures or index.figure_embeddings is None:
        return []
    try:
        clip = _get_clip_model()
        query_vector = clip.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )[0]
        scores = index.figure_embeddings @ query_vector
        ranked = np.argsort(scores)[::-1][:min(limit, len(index.figures))]
        output = []
        for rank_index in ranked:
            figure = index.figures[int(rank_index)]
            output.append({
                "page": figure["page"],
                "score": float(scores[int(rank_index)]),
                "mime_type": figure["mime_type"],
                "image_base64": base64.b64encode(figure["bytes"]).decode("ascii"),
                "bytes": figure["bytes"],
            })
        return output
    except Exception as exc:
        print(f"Figure search skipped: {exc}")
        return []


def _describe_figures(query: str, figures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not figures:
        return []
    _, vision_llm = _get_openai_models()
    from langchain_core.messages import HumanMessage
    descriptions = []
    for figure in figures[:MAX_IMAGES_FOR_VLM]:
        data_url = f"data:{figure['mime_type']};base64,{base64.b64encode(figure['bytes']).decode('ascii')}"
        try:
            message = HumanMessage(content=[
                {
                    "type": "text",
                    "text": (
                        f"Question: {query}\nInspect this research-paper figure carefully. "
                        "Describe only visible content. Read labels, axes, units, dates and values only when legible. "
                        "Do not guess missing or unreadable values. State if the figure does not help answer the question."
                    ),
                },
                {"type": "image_url", "image_url": {"url": data_url}},
            ])
            answer = vision_llm.invoke([message]).content
            descriptions.append({"page": figure["page"], "description": str(answer)})
        except Exception as exc:
            descriptions.append({"page": figure["page"], "description": f"Figure analysis failed: {exc}"})
    return descriptions


def _generate_answer(
    query: str,
    text_results: list[dict[str, Any]],
    table_results: list[dict[str, Any]],
    figure_descriptions: list[dict[str, Any]],
) -> str:
    answer_llm, _ = _get_openai_models()
    import json
    evidence = {
        "text_evidence": [
            {"page": item.get("page"), "text": item.get("text", "")[:1800]}
            for item in text_results
        ],
        "table_evidence": [
            {"page": item.get("page"), "text": item.get("text", "")[:5000]}
            for item in table_results
        ],
        "figure_evidence": figure_descriptions,
    }
    prompt = f"""You are a research-paper assistant. Answer using ONLY the retrieved evidence below.
Do not invent facts, numbers, dates, page references, or conclusions. Include physical PDF page citations
like (PDF p. 3) when evidence gives a page. If the evidence does not support the answer, say so clearly.
Distinguish direct evidence from uncertainty. Use a direct, clear response.

Question: {query}

Retrieved evidence:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Grounded answer:"""
    try:
        result = answer_llm.invoke(prompt)
        return str(result.content)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Answer generation failed: {exc}") from exc


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "Multimodal AI Research Assistant API", "health": "/health", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, Any]:
    # Does not trigger model loading; safe to call as an inexpensive health check.
    return {
        "status": "ok",
        "uptime_seconds": round(time.monotonic() - _process_started_at, 1),
        "process_rss_mb": _memory_usage_mb(),
        "text_model_loaded": _embedding_model is not None,
        "text_model_load_seconds": _embedding_model_load_seconds,
        "clip_model_loaded": _clip_model is not None,
        "clip_model_load_seconds": _clip_model_load_seconds,
        "cached_document_sessions": len(_indexes),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "model_cache_directory": os.getenv("HF_HOME", "Hugging Face default cache"),
        "mode": "lazy-model-load; per-process model reuse; per-session in-memory document index",
    }


@app.post("/api/ask")
def ask_question(
    file: UploadFile = File(..., description="Research paper in PDF format"),
    query: str = Form(..., min_length=3, max_length=2000),
    session_id: str = Form(..., min_length=8, max_length=100),
    x_rag_token: str | None = Header(default=None, alias="X-RAG-Token"),
) -> JSONResponse:
    _check_api_token(x_rag_token)
    filename = file.filename or "research-paper.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Please upload a PDF file.")
    pdf_bytes = file.file.read(MAX_PDF_BYTES + 1)
    file.file.close()
    if not pdf_bytes:
        raise HTTPException(status_code=422, detail="The uploaded PDF is empty.")
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF size limit is 20 MB for this demo.")
    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(status_code=415, detail="The uploaded file does not look like a valid PDF.")
    if not query.strip():
        raise HTTPException(status_code=422, detail="Enter a question about the uploaded paper.")
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,100}", session_id):
        raise HTTPException(status_code=422, detail="Invalid session identifier.")

    index = _get_index(session_id, filename, pdf_bytes)
    text_results = _text_search(index, query.strip())
    table_results = _table_search(index, query.strip())
    figures = _figure_search(index, query.strip())
    try:
        descriptions = _describe_figures(query.strip(), figures)
    except HTTPException:
        raise
    except Exception as exc:
        descriptions = [{"page": item["page"], "description": f"Figure analysis failed: {exc}"} for item in figures[:MAX_IMAGES_FOR_VLM]]
    answer = _generate_answer(query.strip(), text_results, table_results, descriptions)

    safe_figures = [
        {"page": item["page"], "score": item["score"], "mime_type": item["mime_type"], "image_base64": item["image_base64"]}
        for item in figures
    ]
    return JSONResponse({
        "filename": filename,
        "question": query.strip(),
        "answer": answer,
        "text_sources": [
            {"page": item.get("page"), "score": item.get("score"), "text": item.get("text", "")[:1800]}
            for item in text_results
        ],
        "tables": table_results,
        "figures": safe_figures,
        "figure_descriptions": descriptions,
        "index_stats": {
            "text_chunks": sum(item["kind"] == "text" for item in index.documents),
            "tables": sum(item["kind"] == "table" for item in index.documents),
            "figures": len(index.figures),
            "cached_for_session": True,
        },
    })
