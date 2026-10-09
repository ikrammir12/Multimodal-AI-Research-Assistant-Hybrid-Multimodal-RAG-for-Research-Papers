"""FastAPI serving layer for the existing multimodal research-paper RAG index.

Important: this module loads the persisted ChromaDB collections. It does not
import or rerun complete_multimodal_ai_research_assistant.py, which rebuilds the index.
"""
from __future__ import annotations

import base64
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db")))
TEXT_COLLECTION_NAME = "research_papers"
IMAGE_COLLECTION_NAME = "paper_images"
TABLE_COLLECTION_NAME = "paper_tables"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
CLIP_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
TEXT_TOP_K = 5
TABLE_TOP_K = 3
IMAGE_TOP_K = 3
MAX_IMAGES_FOR_VLM = 2

app = FastAPI(
    title="Multimodal AI Research Assistant API",
    version="1.0.0",
    description="Hybrid text, table, and figure retrieval over an indexed research paper.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000, description="Question about the indexed paper")


state: dict[str, Any] = {
    "ready": False,
    "startup_error": None,
    "client": None,
    "text_collection": None,
    "image_collection": None,
    "table_collection": None,
    "all_texts": [],
    "all_ids": [],
    "all_metadata": [],
    "bm25": None,
    "embedding_model": None,
    "clip_model": None,
    "reranker": None,
    "vision_llm": None,
    "answer_llm": None,
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").lower())


def _collection_names(client: chromadb.ClientAPI) -> set[str]:
    return {
        item if isinstance(item, str) else item.name
        for item in client.list_collections()
    }


@app.on_event("startup")
def load_rag_resources() -> None:
    """Load the existing vector collections and query-time models once."""
    try:
        if not CHROMA_DIR.exists():
            raise RuntimeError(
                f"ChromaDB folder not found: {CHROMA_DIR}. Run the main indexing pipeline first."
            )
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        required = {TEXT_COLLECTION_NAME, IMAGE_COLLECTION_NAME, TABLE_COLLECTION_NAME}
        missing = required - _collection_names(client)
        if missing:
            raise RuntimeError(
                "Missing ChromaDB collection(s): " + ", ".join(sorted(missing))
                + ". Run complete_multimodal_ai_research_assistant.py first."
            )

        text_collection = client.get_collection(TEXT_COLLECTION_NAME)
        image_collection = client.get_collection(IMAGE_COLLECTION_NAME)
        table_collection = client.get_collection(TABLE_COLLECTION_NAME)
        text_data = text_collection.get(include=["documents", "metadatas"])
        all_texts = text_data.get("documents") or []
        all_ids = text_data.get("ids") or []
        all_metadata = text_data.get("metadatas") or []
        if not all_texts:
            raise RuntimeError("The research_papers collection is empty. Rebuild the index first.")

        print(f"Loading {len(all_texts)} text chunks from {CHROMA_DIR}...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

        bm25 = None
        try:
            from rank_bm25 import BM25Okapi
            bm25 = BM25Okapi([tokenize(text) for text in all_texts])
        except Exception as exc:
            print(f"BM25 unavailable; dense retrieval will still work. Reason: {exc}")

        reranker = None
        try:
            from FlagEmbedding import FlagReranker
            reranker = FlagReranker(RERANKER_MODEL_NAME, use_fp16=False, devices=["cpu"])
        except Exception as exc:
            print(f"Reranker unavailable; using Reciprocal Rank Fusion only. Reason: {exc}")

        clip_model = None
        if image_collection.count() > 0:
            print(f"Loading CLIP for {image_collection.count()} indexed figures...")
            clip_model = SentenceTransformer(CLIP_MODEL_NAME, device="cpu")

        api_key = os.getenv("OPENAI_API_KEY")
        vision_llm = answer_llm = None
        if api_key:
            from langchain_openai import ChatOpenAI
            vision_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
            answer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=api_key)
        else:
            print("OPENAI_API_KEY not found; retrieval will load, but /api/query will return HTTP 503 until configured.")

        state.update({
            "ready": True,
            "client": client,
            "text_collection": text_collection,
            "image_collection": image_collection,
            "table_collection": table_collection,
            "all_texts": all_texts,
            "all_ids": all_ids,
            "all_metadata": all_metadata,
            "bm25": bm25,
            "embedding_model": embedding_model,
            "clip_model": clip_model,
            "reranker": reranker,
            "vision_llm": vision_llm,
            "answer_llm": answer_llm,
        })
        print("RAG API startup complete.")
    except Exception as exc:
        state["startup_error"] = str(exc)
        print(f"RAG API startup error: {exc}")


def _page(result: dict[str, Any]) -> int | None:
    try:
        value = (result.get("metadata") or {}).get("page")
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _dense_search(query: str, limit: int = 10) -> list[dict[str, Any]]:
    collection = state["text_collection"]
    emb = state["embedding_model"].encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )[0]
    count = collection.count()
    if not count:
        return []
    result = collection.query(
        query_embeddings=[emb.tolist()],
        n_results=min(limit, count),
        include=["documents", "metadatas", "distances"],
    )
    items = []
    for item_id, document, metadata, distance in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        items.append({
            "id": item_id,
            "document": document or "",
            "metadata": metadata or {},
            "score": 1.0 - float(distance),
            "distance": float(distance),
        })
    return items


def _hybrid_search(query: str, limit: int = TEXT_TOP_K) -> list[dict[str, Any]]:
    dense = _dense_search(query, limit=10)
    all_texts = state["all_texts"]
    all_ids = state["all_ids"]
    all_metadata = state["all_metadata"]
    bm25 = state["bm25"]

    sparse: list[dict[str, Any]] = []
    if bm25 is not None:
        scores = np.asarray(bm25.get_scores(tokenize(query)))
        top_indices = np.argsort(scores)[::-1][:min(10, len(all_texts))]
        sparse = [
            {
                "id": all_ids[int(index)],
                "document": all_texts[int(index)] or "",
                "metadata": all_metadata[int(index)] or {},
                "score": float(scores[int(index)]),
            }
            for index in top_indices
        ]

    by_id = {
        item_id: {"id": item_id, "document": doc or "", "metadata": metadata or {}}
        for item_id, doc, metadata in zip(all_ids, all_texts, all_metadata)
    }
    rrf_scores: dict[str, float] = {}
    for ranked in (dense, sparse):
        for rank, item in enumerate(ranked, start=1):
            rrf_scores[item["id"]] = rrf_scores.get(item["id"], 0.0) + 1.0 / (60 + rank)
    candidate_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:10]
    candidates = [by_id[item_id].copy() for item_id in candidate_ids if item_id in by_id]
    for item in candidates:
        item["rrf_score"] = rrf_scores[item["id"]]

    reranker = state["reranker"]
    if reranker is not None and candidates:
        try:
            scores = reranker.compute_score([[query, item["document"]] for item in candidates], normalize=True)
            if isinstance(scores, (float, int)):
                scores = [scores]
            for item, score in zip(candidates, scores):
                item["reranker_score"] = float(score)
            candidates.sort(key=lambda item: item.get("reranker_score", -1), reverse=True)
        except Exception as exc:
            print(f"Reranking failed for this request; falling back to RRF. Reason: {exc}")
            candidates.sort(key=lambda item: item["rrf_score"], reverse=True)
    else:
        candidates.sort(key=lambda item: item["rrf_score"], reverse=True)
    return candidates[:limit]


def _table_search(query: str, limit: int = TABLE_TOP_K) -> list[dict[str, Any]]:
    collection = state["table_collection"]
    if collection.count() == 0:
        return []
    emb = state["embedding_model"].encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )[0]
    result = collection.query(
        query_embeddings=[emb.tolist()],
        n_results=min(limit, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "page": (metadata or {}).get("page"),
            "text": document or "",
            "score": 1.0 - float(distance),
        }
        for document, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def _figure_search(query: str, limit: int = IMAGE_TOP_K) -> list[dict[str, Any]]:
    collection = state["image_collection"]
    model = state["clip_model"]
    if model is None or collection.count() == 0:
        return []
    emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    result = collection.query(
        query_embeddings=[emb.tolist()],
        n_results=min(limit, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    items = []
    for document, metadata, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        raw_path = str(document or (metadata or {}).get("source") or "")
        raw_path = raw_path.replace("\\", os.sep)
        path = Path(raw_path)
        if not path.is_absolute():
            path = (BASE_DIR / path).resolve()
        items.append({
            "page": (metadata or {}).get("page"),
            "image_path": str(path),
            "exists": path.is_file(),
            "score": 1.0 - float(distance),
        })
    return items


def _image_to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _describe_figures(query: str, figures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    llm = state["vision_llm"]
    if llm is None:
        return []
    from langchain_core.messages import HumanMessage
    descriptions = []
    for figure in figures[:MAX_IMAGES_FOR_VLM]:
        path = Path(figure["image_path"])
        if not path.is_file():
            continue
        try:
            message = HumanMessage(content=[
                {"type": "text", "text": (
                    f"Question: {query}\nInspect this research-paper figure. First say if it is relevant. "
                    "Report the title, axes, labels, units, dates, legend and numbers only if they are visible. "
                    "Do not infer unreadable values; say clearly if a value cannot be read."
                )},
                {"type": "image_url", "image_url": {"url": _image_to_data_url(path)}},
            ])
            description = llm.invoke([message]).content
            descriptions.append({"page": figure["page"], "image_path": str(path), "description": description})
        except Exception as exc:
            descriptions.append({"page": figure["page"], "image_path": str(path), "description": f"VLM_ERROR: {exc}"})
    return descriptions


def _answer(query: str, text_results: list[dict[str, Any]], tables: list[dict[str, Any]], figures: list[dict[str, Any]], descriptions: list[dict[str, Any]]) -> str:
    llm = state["answer_llm"]
    if llm is None:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is missing. Add it to the project's .env file and restart the API.")
    evidence = {
        "text_evidence": [
            {"page": _page(item), "text": item.get("document", "")[:1800]}
            for item in text_results
        ],
        "table_evidence": [
            {"page": item["page"], "text": item["text"][:3500]}
            for item in tables
        ],
        "figure_evidence": descriptions,
    }
    prompt = f"""You are a research-paper assistant. Answer using ONLY the retrieved evidence below.
Do not invent facts, numbers, dates, page references, or conclusions. Include physical PDF page citations
like (PDF p. 3) when evidence gives a page. If sources do not support the answer, state that limitation.
Use a direct, clear response and distinguish explicit evidence from uncertainty.

Question: {query}

Retrieved evidence:
{__import__('json').dumps(evidence, ensure_ascii=False, indent=2)}

Grounded answer:"""
    try:
        response = llm.invoke(prompt)
        return str(response.content)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI answer generation failed: {exc}") from exc


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if state["ready"] else "not_ready",
        "chroma_dir": str(CHROMA_DIR),
        "text_chunks": len(state["all_texts"]),
        "figures": state["image_collection"].count() if state["image_collection"] is not None else 0,
        "tables": state["table_collection"].count() if state["table_collection"] is not None else 0,
        "startup_error": state["startup_error"],
        "openai_configured": state["answer_llm"] is not None,
    }


@app.post("/api/query")
def query_paper(request: QueryRequest) -> dict[str, Any]:
    if not state["ready"]:
        raise HTTPException(status_code=503, detail=state["startup_error"] or "RAG resources are still loading.")
    if state["answer_llm"] is None:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is missing. Add it to .env and restart the API.")

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Enter a non-empty question.")

    text_results = _hybrid_search(query)
    tables = _table_search(query)
    figures = _figure_search(query)
    descriptions = _describe_figures(query, figures)
    answer = _answer(query, text_results, tables, figures, descriptions)

    return {
        "question": query,
        "answer": answer,
        "text_sources": [
            {"page": _page(item), "score": item.get("reranker_score", item.get("rrf_score")),
             "text": item.get("document", "")[:1800]}
            for item in text_results
        ],
        "tables": tables,
        "figures": figures,
        "figure_descriptions": descriptions,
    }
