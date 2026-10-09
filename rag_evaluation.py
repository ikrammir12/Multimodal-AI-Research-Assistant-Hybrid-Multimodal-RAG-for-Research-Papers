"""Retrieval evaluation for the Multimodal AI Research Assistant.

This script reads the ChromaDB collections created by

complete_multimodal_ai_research_assistant.py. It does NOT call OpenAI and

therefore can evaluate retrieval even when the VLM API connection is down.

Run from the project root:

    uv run python rag_evaluation.py

First annotate rag_eval_dataset.json with relevant physical PDF page numbers.

"""

import base64

import csv

import json

import mimetypes

import os

import re

import statistics

from pathlib import Path

import chromadb

import numpy as np

from FlagEmbedding import FlagReranker

from sentence_transformers import SentenceTransformer

# --------------------------- CONFIG ---------------------------

PROJECT_DIR = Path.cwd()

CHROMA_DIR = PROJECT_DIR / "chroma_db"

DATASET_PATH = PROJECT_DIR / "rag_eval_dataset_annotated.json"

RESULTS_CSV = PROJECT_DIR / "rag_evaluation_results.csv"

SUMMARY_JSON = PROJECT_DIR / "rag_evaluation_summary.json"

TEXT_COLLECTION = "research_papers"

IMAGE_COLLECTION = "paper_images"

TABLE_COLLECTION = "paper_tables"

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

CLIP_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

EVAL_TOP_K = 5

DENSE_CANDIDATE_K = 10

BM25_CANDIDATE_K = 10

HYBRID_CANDIDATE_K = 10

USE_RERANKER = True  # If it cannot load, evaluation falls back to RRF ranking.

RUN_ANSWER_GENERATION = True  # Generate grounded answers using the OpenAI API.
MAX_ANSWER_QUESTIONS = 3  # Generate answers only for the first 3 questions during the initial test.
MAX_IMAGES_FOR_ANSWER = 1  # Limit image-analysis API calls during the initial test.

# ------------------------- HELPERS ---------------------------

def tokenize(text):

    return re.findall(r"\w+", (text or "").lower())

def get_page(result):

    """Return a physical PDF page number, or None if metadata lacks it."""

    metadata = result.get("metadata") or {}

    page = metadata.get("page")

    try:

        return int(page) if page is not None and int(page) > 0 else None

    except (TypeError, ValueError):

        return None

def unique_pages(results):

    pages = []

    for result in results:

        page = get_page(result)

        if page is not None and page not in pages:

            pages.append(page)

    return pages

def parse_page_labels(value):

    if not value:

        return []

    parsed = []

    for page in value:

        try:

            parsed.append(int(page))

        except (TypeError, ValueError):

            continue

    return sorted(set(page for page in parsed if page > 0))

def retrieval_metrics(results, relevant_pages, k=EVAL_TOP_K):

    """Page-level Hit@K, Precision@K, Recall@K and MRR@K.

    Ground truth is page-level, so repeated chunks from the same page count once.

    If no relevant pages have been annotated, metrics are left blank (None).

    """

    relevant = set(parse_page_labels(relevant_pages))

    if not relevant:

        return {

            f"hit@{k}": None,

            f"precision@{k}": None,

            f"recall@{k}": None,

            f"mrr@{k}": None,

        }

    ranked_pages = unique_pages(results)[:k]

    hits = [page for page in ranked_pages if page in relevant]

    first_rank = next(

        (rank for rank, page in enumerate(ranked_pages, start=1) if page in relevant),

        None,

    )

    return {

        f"hit@{k}": int(bool(hits)),

        f"precision@{k}": len(hits) / k,

        f"recall@{k}": len(set(hits)) / len(relevant),

        f"mrr@{k}": 1.0 / first_rank if first_rank else 0.0,

    }

def result_record(item_id, document, metadata=None, score=None, score_name=None):

    record = {

        "id": item_id,

        "document": document or "",

        "metadata": metadata or {},

    }

    if score is not None:

        record[score_name or "score"] = float(score)

    return record

def rows_to_display(results):

    """Compact serializable evidence for the CSV report."""

    displayed = []

    for rank, result in enumerate(results[:EVAL_TOP_K], start=1):

        displayed.append({

            "rank": rank,

            "id": result.get("id"),

            "page": get_page(result),

            "source": (result.get("metadata") or {}).get("source"),

            "score": result.get("score"),

            "distance": result.get("distance"),

            "document": result.get("document", ""),

        })

    return displayed

def metric_columns(prefix, metrics):

    return {f"{prefix}_{name}": value for name, value in metrics.items()}

def aggregate_metric_rows(row_metrics):

    """Average only annotated queries for each metric."""

    summary = {}

    for metric_name in sorted({key for row in row_metrics for key in row}):

        values = [row[metric_name] for row in row_metrics if row.get(metric_name) is not None]

        summary[metric_name] = {

            "evaluated_questions": len(values),

            "mean": round(statistics.mean(values), 4) if values else None,

        }

    return summary

# ----------------------- LOAD DATA ----------------------------

if not DATASET_PATH.is_file():

    raise FileNotFoundError(

        f"Evaluation dataset not found: {DATASET_PATH}. "

        "Place rag_eval_dataset.json beside this script or edit DATASET_PATH."

    )

with DATASET_PATH.open("r", encoding="utf-8") as file:

    dataset_payload = json.load(file)

questions = dataset_payload["questions"] if isinstance(dataset_payload, dict) else dataset_payload

if not questions:

    raise ValueError("The evaluation dataset contains no questions.")

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

required_collections = [TEXT_COLLECTION, IMAGE_COLLECTION, TABLE_COLLECTION]

existing_names = {

    collection if isinstance(collection, str) else collection.name

    for collection in client.list_collections()

}

missing = [name for name in required_collections if name not in existing_names]

if missing:

    raise RuntimeError(

        "Missing ChromaDB collection(s): " + ", ".join(missing) + ". "

        "Run the main pipeline through the indexing stages first, and run this "

        "script from the project root so it uses the same ./chroma_db folder."

    )

text_collection = client.get_collection(TEXT_COLLECTION)

image_collection = client.get_collection(IMAGE_COLLECTION)

table_collection = client.get_collection(TABLE_COLLECTION)

text_data = text_collection.get(include=["documents", "metadatas"])

all_texts = text_data.get("documents") or []

all_ids = text_data.get("ids") or []

all_metadata = text_data.get("metadatas") or []

if not all_texts:

    raise RuntimeError("The research_papers collection is empty.")

print(f"Loaded {len(all_texts)} text chunks from ChromaDB.")

print(f"Loaded {image_collection.count()} figures and {table_collection.count()} tables.")

print("Loading BGE-M3 for query embeddings...")

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

bm25 = None

try:

    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([tokenize(text) for text in all_texts])

except Exception as exc:

    print(f"Warning: BM25 could not be initialized: {exc}")

reranker = None

if USE_RERANKER:

    try:

        print("Loading reranker (if unavailable, RRF ranking will be used)...")

        reranker = FlagReranker(RERANKER_MODEL_NAME, use_fp16=False)

    except Exception as exc:

        print(f"Warning: reranker unavailable; using RRF ranking. Reason: {exc}")

clip_model = None

if image_collection.count() > 0:

    print("Loading CLIP for figure retrieval...")

    clip_model = SentenceTransformer(CLIP_MODEL_NAME)

def dense_search(query, limit=DENSE_CANDIDATE_K):

    query_embedding = embedding_model.encode(

        [query], convert_to_numpy=True, normalize_embeddings=True

    )[0]

    response = text_collection.query(

        query_embeddings=[query_embedding.tolist()],

        n_results=min(limit, text_collection.count()),

        include=["documents", "metadatas", "distances"],

    )

    results = []

    for item_id, document, metadata, distance in zip(

        response["ids"][0],

        response["documents"][0],

        response["metadatas"][0],

        response["distances"][0],

    ):

        results.append(result_record(item_id, document, metadata, score=1.0 - float(distance)))

        results[-1]["distance"] = float(distance)

    return results

def bm25_search(query, limit=BM25_CANDIDATE_K):

    if bm25 is None:

        return []

    scores = np.asarray(bm25.get_scores(tokenize(query)))

    indices = np.argsort(scores)[::-1][:min(limit, len(all_texts))]

    return [

        result_record(all_ids[index], all_texts[index], all_metadata[index], score=scores[index])

        for index in indices

    ]

def hybrid_search(query, limit=EVAL_TOP_K):

    dense = dense_search(query, DENSE_CANDIDATE_K)

    sparse = bm25_search(query, BM25_CANDIDATE_K)

    by_id = {

        item_id: result_record(item_id, doc, metadata)

        for item_id, doc, metadata in zip(all_ids, all_texts, all_metadata)

    }

    rrf_scores = {}

    for rank, item in enumerate(dense, start=1):

        rrf_scores[item["id"]] = rrf_scores.get(item["id"], 0.0) + 1.0 / (60 + rank)

    for rank, item in enumerate(sparse, start=1):

        rrf_scores[item["id"]] = rrf_scores.get(item["id"], 0.0) + 1.0 / (60 + rank)

    candidate_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:HYBRID_CANDIDATE_K]

    candidates = [by_id[item_id] for item_id in candidate_ids if item_id in by_id]

    for item in candidates:

        item["rrf_score"] = rrf_scores[item["id"]]

    if reranker is not None and candidates:

        pairs = [[query, item["document"]] for item in candidates]

        scores = reranker.compute_score(pairs, normalize=True)

        if isinstance(scores, (float, int)):

            scores = [scores]

        for item, score in zip(candidates, scores):

            item["reranker_score"] = float(score)

        candidates.sort(key=lambda item: item.get("reranker_score", -1), reverse=True)

    else:

        candidates.sort(key=lambda item: item["rrf_score"], reverse=True)

    return candidates[:limit]

def table_search(query, limit=EVAL_TOP_K):

    if table_collection.count() == 0:

        return []

    query_embedding = embedding_model.encode(

        [query], convert_to_numpy=True, normalize_embeddings=True

    )[0]

    response = table_collection.query(

        query_embeddings=[query_embedding.tolist()],

        n_results=min(limit, table_collection.count()),

        include=["documents", "metadatas", "distances"],

    )

    results = []

    for item_id, document, metadata, distance in zip(

        response["ids"][0], response["documents"][0], response["metadatas"][0], response["distances"][0]

    ):

        record = result_record(item_id, document, metadata, score=1.0 - float(distance))

        record["distance"] = float(distance)

        results.append(record)

    return results

def figure_search(query, limit=EVAL_TOP_K):

    if clip_model is None or image_collection.count() == 0:

        return []

    # SentenceTransformers' CLIP model supports text inputs using the same shared space.

    query_embedding = clip_model.encode(

        [query], convert_to_numpy=True, normalize_embeddings=True

    )[0]

    response = image_collection.query(

        query_embeddings=[query_embedding.tolist()],

        n_results=min(limit, image_collection.count()),

        include=["documents", "metadatas", "distances"],

    )

    results = []

    for item_id, document, metadata, distance in zip(

        response["ids"][0], response["documents"][0], response["metadatas"][0], response["distances"][0]

    ):

        record = result_record(item_id, document, metadata, score=1.0 - float(distance))

        record["distance"] = float(distance)

        results.append(record)

    return results

# -------------------- OPTIONAL ANSWER EVALUATION ---------------

# Retrieval scoring works without the OpenAI API. Enable this only when the

# network/API call is working; the initial run generates answers for the first few questions.

vision_llm = None

answer_llm = None

if RUN_ANSWER_GENERATION:

    from dotenv import load_dotenv

    from langchain_core.messages import HumanMessage

    from langchain_openai import ChatOpenAI

    load_dotenv(PROJECT_DIR / ".env")

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:

        raise RuntimeError("RUN_ANSWER_GENERATION=True, but OPENAI_API_KEY is missing.")

    vision_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

    answer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=api_key)

def image_to_data_url(image_path):

    mime_type, _ = mimetypes.guess_type(str(image_path))

    mime_type = mime_type or "image/png"

    with open(image_path, "rb") as image_file:

        encoded = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"

def generate_grounded_answer(query, text_results, tables_results, figures_results):

    """Run VLM descriptions and the final answer, returning text plus context."""

    if vision_llm is None or answer_llm is None:

        return "", [], "Answer generation is disabled. Set RUN_ANSWER_GENERATION=True to enable it."

    figure_descriptions = []

    for result in figures_results[:MAX_IMAGES_FOR_ANSWER]:

        image_path = result.get("document", "")

        if not image_path or not Path(image_path).is_file():

            continue

        try:

            message = HumanMessage(content=[

                {

                    "type": "text",

                    "text": (

                        f"Question: {query}\n"

                        "Inspect this research-paper figure carefully. Describe only what is visible. "

                        "Read labels and values only if legible; do not guess missing values. "

                        "Say when the figure does not help answer the question."

                    ),

                },

                {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},

            ])

            response = vision_llm.invoke([message])

            figure_descriptions.append({

                "page": get_page(result),

                "source": image_path,

                "description": response.content,

            })

        except Exception as exc:

            figure_descriptions.append({

                "page": get_page(result),

                "source": image_path,

                "description": f"VLM_ERROR: {exc}",

            })

    context = {

        "text_evidence": [

            {"page": get_page(result), "text": result.get("document", "")}

            for result in text_results[:EVAL_TOP_K]

        ],

        "table_evidence": [

            {"page": get_page(result), "text": result.get("document", "")}

            for result in tables_results[:EVAL_TOP_K]

        ],

        "figure_evidence": figure_descriptions,

    }

    prompt = f"""You are a research-paper RAG assistant. Answer using ONLY the evidence below.

Do not invent values, dates, page references, or conclusions. If the evidence is incomplete,

say what cannot be established. Distinguish direct evidence from uncertainty and include

physical page numbers where provided.

Question:

{query}

Retrieved evidence:

{json.dumps(context, ensure_ascii=False, indent=2)}

Grounded answer:

"""

    try:

        answer = answer_llm.invoke(prompt).content

        return answer, figure_descriptions, ""

    except Exception as exc:

        return "", figure_descriptions, f"ANSWER_API_ERROR: {exc}"

# ------------------------- EVALUATE ---------------------------

output_rows = []

aggregate_by_method = {

    name: [] for name in ["dense", "bm25", "hybrid", "tables", "figures"]

}

for index, item in enumerate(questions, start=1):

    question_id = item.get("id", f"Q{index:02d}")

    query = item.get("question", "").strip()

    if not query:

        print(f"Skipping {question_id}: empty question.")

        continue

    print(f"[{index}/{len(questions)}] {question_id}: {query}")

    dense_results = dense_search(query, DENSE_CANDIDATE_K)

    sparse_results = bm25_search(query, BM25_CANDIDATE_K)

    hybrid_results = hybrid_search(query, EVAL_TOP_K)

    tables_results = table_search(query, EVAL_TOP_K)

    figures_results = figure_search(query, EVAL_TOP_K)

    text_gt = item.get("relevant_text_pages", [])

    table_gt = item.get("relevant_table_pages", [])

    figure_gt = item.get("relevant_figure_pages", [])

    generated_answer = ""

    answer_generation_error = ""

    generated_figure_descriptions = []

    if RUN_ANSWER_GENERATION and index <= MAX_ANSWER_QUESTIONS:

        generated_answer, generated_figure_descriptions, answer_generation_error = generate_grounded_answer(

            query, hybrid_results, tables_results, figures_results

        )

    elif RUN_ANSWER_GENERATION:

        answer_generation_error = (

            f"Skipped to limit API usage: only the first {MAX_ANSWER_QUESTIONS} questions "

            "are configured for answer generation. Increase MAX_ANSWER_QUESTIONS to generate more."

        )

    metrics = {

        "dense": retrieval_metrics(dense_results, text_gt),

        "bm25": retrieval_metrics(sparse_results, text_gt),

        "hybrid": retrieval_metrics(hybrid_results, text_gt),

        "tables": retrieval_metrics(tables_results, table_gt),

        "figures": retrieval_metrics(figures_results, figure_gt),

    }

    for method, method_metrics in metrics.items():

        aggregate_by_method[method].append(method_metrics)

    row = {

        "id": question_id,

        "type": item.get("type", ""),

        "question": query,

        "reference_answer": item.get("reference_answer", ""),

        "generated_answer": generated_answer,

        "answer_generation_error": answer_generation_error,

        "generated_figure_descriptions": json.dumps(generated_figure_descriptions, ensure_ascii=False),

        "relevant_text_pages": json.dumps(parse_page_labels(text_gt)),

        "relevant_table_pages": json.dumps(parse_page_labels(table_gt)),

        "relevant_figure_pages": json.dumps(parse_page_labels(figure_gt)),

        "dense_top_pages": json.dumps(unique_pages(dense_results)),

        "bm25_top_pages": json.dumps(unique_pages(sparse_results)),

        "hybrid_top_pages": json.dumps(unique_pages(hybrid_results)),

        "table_top_pages": json.dumps(unique_pages(tables_results)),

        "figure_top_pages": json.dumps(unique_pages(figures_results)),

        "dense_evidence": json.dumps(rows_to_display(dense_results), ensure_ascii=False),

        "bm25_evidence": json.dumps(rows_to_display(sparse_results), ensure_ascii=False),

        "hybrid_evidence": json.dumps(rows_to_display(hybrid_results), ensure_ascii=False),

        "table_evidence": json.dumps(rows_to_display(tables_results), ensure_ascii=False),

        "figure_evidence": json.dumps(rows_to_display(figures_results), ensure_ascii=False),

        # Fill these after reviewing the model's final answer for this question.

        "answer_correctness_0_to_2": "",

        "answer_groundedness_0_to_2": "",

        "answer_completeness_0_to_2": "",

        "citation_accuracy_0_to_2": "",

        "review_notes": "",

    }

    for method, method_metrics in metrics.items():

        row.update(metric_columns(method, method_metrics))

    output_rows.append(row)

# Write detailed evidence and metrics.

if output_rows:

    columns = list(dict.fromkeys(key for row in output_rows for key in row.keys()))

    with RESULTS_CSV.open("w", newline="", encoding="utf-8-sig") as file:

        writer = csv.DictWriter(file, fieldnames=columns)

        writer.writeheader()

        writer.writerows(output_rows)

summary = {

    "dataset": str(DATASET_PATH),

    "results_csv": str(RESULTS_CSV),

    "top_k": EVAL_TOP_K,

    "note": (

        "Page-level retrieval metrics are computed only for questions with non-empty relevant page labels. "

        "Annotate the JSON ground truth and rerun to obtain meaningful aggregate scores."

    ),

    "methods": {

        method: aggregate_metric_rows(rows)

        for method, rows in aggregate_by_method.items()

    },

}

with SUMMARY_JSON.open("w", encoding="utf-8") as file:

    json.dump(summary, file, ensure_ascii=False, indent=2)

print("\nEvaluation finished.")

print(f"Detailed evidence and metrics: {RESULTS_CSV}")

print(f"Aggregate summary: {SUMMARY_JSON}")

print("Metrics are blank for any modality/question whose relevant pages are not annotated.")

if RUN_ANSWER_GENERATION:

    print(f"Answer generation was enabled for the first {MAX_ANSWER_QUESTIONS} questions; review generated_answer against reference_answer and fill the manual score columns.")

else:

    print("Answer generation is disabled; set RUN_ANSWER_GENERATION=True after the OpenAI API connection works.")

print("No answer scores are automatic: fill the manual score columns after comparing answers with the PDF.")
