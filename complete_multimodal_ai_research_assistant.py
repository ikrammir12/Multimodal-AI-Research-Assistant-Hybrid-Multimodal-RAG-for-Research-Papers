"""Complete Hybrid Multimodal RAG pipeline for research-paper PDFs.

Pipeline:
PDF text -> chunks -> BGE-M3 + BM25 -> hybrid retrieval -> reranking
PDF figures -> placement/size filtering -> CLIP -> ChromaDB -> figure retrieval
PDF tables -> extraction -> BGE-M3 -> ChromaDB -> table retrieval
Retrieved evidence -> VLM figure analysis -> final grounded LLM answer

Before running, update PDF_PATH and set OPENAI_API_KEY in a .env file.
"""

import base64
import json
import os
import re
from pathlib import Path

import chromadb
import numpy as np
import pymupdf
from dotenv import load_dotenv
from FlagEmbedding import FlagReranker
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================
PDF_PATH = r"D:\CV project\Multimodal AI Research Assistant\Multimodal-AI-Research-Assistant-Hybrid-Multimodal-RAG-for-Research-Papers\12_Population.pdf"
IMAGE_DIR = "extracted_images"
CHROMA_DIR = "./chroma_db"
TEXT_COLLECTION_NAME = "research_papers"
IMAGE_COLLECTION_NAME = "paper_images"
TABLE_COLLECTION_NAME = "paper_tables"

QUERY = "Tell me about the growth rate of Pakistan?"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
DENSE_TOP_K = 8
SPARSE_TOP_K = 8
HYBRID_CANDIDATE_K = 10
FINAL_TEXT_TOP_K = 5
TOP_K_IMAGES = 5
TOP_K_TABLES = 5
MAX_IMAGES_FOR_VLM = 3

# Image filters. Adjust these only if valid figures are being removed.
MIN_IMAGE_WIDTH = 64
MIN_IMAGE_HEIGHT = 64
MIN_IMAGE_PIXELS = 10_000
MIN_ASPECT_RATIO = 0.10
MAX_ASPECT_RATIO = 10.0


# ============================================================
# ENVIRONMENT AND CLIENTS
# ============================================================
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv()  # Also check the current working directory.

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY was not found. Add it to a .env file before running."
    )

pdf_file = Path(PDF_PATH)
if not pdf_file.is_file():
    raise FileNotFoundError(
        f"PDF not found: {PDF_PATH}\nUpdate PDF_PATH at the top of this script."
    )

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)


# ============================================================
# HELPER: REBUILD A DEDICATED CHROMADB COLLECTION
# ============================================================
def rebuild_collection(client, collection_name):
    """Replace a pipeline-owned collection to remove stale records."""
    existing = client.list_collections()
    existing_names = {
        item if isinstance(item, str) else item.name
        for item in existing
    }
    if collection_name in existing_names:
        client.delete_collection(name=collection_name)

    return client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


# ============================================================
# 1. EXTRACT TEXT AND SPLIT IT INTO PAGE-AWARE CHUNKS
# ============================================================
def extract_text_chunks(pdf_path, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Extract each PDF page and split it while preserving its page number."""
    reader = PdfReader(str(pdf_path))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    chunks = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if not page_text:
            continue

        for text in splitter.split_text(page_text):
            chunks.append({"text": text, "page": page_number})

    if not chunks:
        raise RuntimeError(
            "No text could be extracted from this PDF. It may be scanned; OCR may be needed."
        )

    return chunks


print("\n[1/9] EXTRACTING PDF TEXT AND CREATING CHUNKS...")
text_chunks = extract_text_chunks(PDF_PATH)
chunk_texts = [item["text"] for item in text_chunks]
print(f"Text chunks created: {len(text_chunks)}")


# ============================================================
# 2. CREATE BGE-M3 EMBEDDINGS AND STORE TEXT IN CHROMADB
# ============================================================
print("\n[2/9] CREATING BGE-M3 TEXT EMBEDDINGS...")
embedding_model = SentenceTransformer("BAAI/bge-m3")
text_embeddings = embedding_model.encode(
    chunk_texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)
print("Text embedding shape:", text_embeddings.shape)

# The collection names in this script are treated as dedicated to this PDF pipeline.
# Rebuilding removes stale chunks/tables/figures from previous runs.
text_collection = rebuild_collection(chroma_client, TEXT_COLLECTION_NAME)
text_collection.upsert(
    ids=[f"text_chunk_{i}" for i in range(len(text_chunks))],
    documents=chunk_texts,
    embeddings=text_embeddings.tolist(),
    metadatas=[
        {
            "type": "text",
            "source_pdf": pdf_file.name,
            "page": item["page"],
        }
        for item in text_chunks
    ],
)
print("Text chunks stored in ChromaDB:", text_collection.count())


# ============================================================
# 3. HYBRID RETRIEVAL: DENSE BGE-M3 + SPARSE BM25 + RERANKER
# ============================================================
def tokenize(text):
    return re.findall(r"\w+", text.lower())


print("\n[3/9] BUILDING BM25 INDEX AND LOADING RERANKER...")
bm25 = BM25Okapi([tokenize(text) for text in chunk_texts])

try:
    reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)
except Exception as exc:
    # The pipeline can still run with hybrid RRF ranking if the reranker
    # model cannot be downloaded or loaded on this machine.
    print(f"Reranker unavailable; using hybrid ranking without reranking.\nReason: {exc}")
    reranker = None


def hybrid_retrieval(query, candidate_k=HYBRID_CANDIDATE_K, final_k=FINAL_TEXT_TOP_K):
    """Combine dense and BM25 ranks with reciprocal rank fusion, then rerank."""
    if not text_chunks:
        return []

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    dense_scores = np.dot(text_embeddings, query_embedding)
    dense_indices = np.argsort(dense_scores)[::-1][:min(DENSE_TOP_K, len(text_chunks))]

    sparse_scores = np.asarray(bm25.get_scores(tokenize(query)))
    sparse_indices = np.argsort(sparse_scores)[::-1][:min(SPARSE_TOP_K, len(text_chunks))]

    # Reciprocal Rank Fusion combines the independent dense and sparse rankings.
    rrf_scores = {}
    for rank, index in enumerate(dense_indices, start=1):
        rrf_scores[int(index)] = rrf_scores.get(int(index), 0.0) + 1.0 / (60 + rank)
    for rank, index in enumerate(sparse_indices, start=1):
        rrf_scores[int(index)] = rrf_scores.get(int(index), 0.0) + 1.0 / (60 + rank)

    candidate_indices = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True,
    )[:min(candidate_k, len(rrf_scores))]

    candidates = [
        {
            "text": text_chunks[index]["text"],
            "page": text_chunks[index]["page"],
            "rrf_score": rrf_scores[index],
            "dense_score": float(dense_scores[index]),
            "bm25_score": float(sparse_scores[index]),
        }
        for index in candidate_indices
    ]

    if reranker is not None and candidates:
        pairs = [[query, item["text"]] for item in candidates]
        scores = reranker.compute_score(pairs, normalize=True)
        if isinstance(scores, (int, float)):
            scores = [scores]
        for item, score in zip(candidates, scores):
            item["reranker_score"] = float(score)
        candidates.sort(
            key=lambda item: item["reranker_score"],
            reverse=True,
        )
    else:
        candidates.sort(key=lambda item: item["rrf_score"], reverse=True)

    return candidates[:min(final_k, len(candidates))]


text_results = hybrid_retrieval(QUERY)
print(f"\nTop {len(text_results)} text passages for: {QUERY}")
for rank, item in enumerate(text_results, start=1):
    print(f"\n{rank}. Page {item['page']}")
    print(item["text"])
    if "reranker_score" in item:
        print(f"Reranker score: {item['reranker_score']:.4f}")


# ============================================================
# 4. EXTRACT, VALIDATE, AND FILTER PDF IMAGES
# ============================================================
def extract_pdf_images(
    pdf_path,
    output_dir=IMAGE_DIR,
    min_width=MIN_IMAGE_WIDTH,
    min_height=MIN_IMAGE_HEIGHT,
    min_pixels=MIN_IMAGE_PIXELS,
    min_aspect_ratio=MIN_ASPECT_RATIO,
    max_aspect_ratio=MAX_ASPECT_RATIO,
):
    """Extract only placed images that pass basic dimension/shape filters."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    total_objects = 0
    skipped_no_placement = 0
    skipped_small = 0
    skipped_aspect = 0
    extraction_errors = 0

    with pymupdf.open(str(pdf_path)) as doc:
        for page_index, page in enumerate(doc, start=1):
            page_images = page.get_images(full=True)

            for image_index, image_info in enumerate(page_images, start=1):
                total_objects += 1
                xref = image_info[0]

                try:
                    # Change 1: ignore image objects without a page placement.
                    rects = page.get_image_rects(xref)
                    if not rects:
                        skipped_no_placement += 1
                        continue

                    pix = pymupdf.Pixmap(doc, xref)
                    if pix.n - pix.alpha > 3:
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

                    width, height = pix.width, pix.height
                    pixels = width * height

                    # Change 2: remove tiny/low-resolution images.
                    if (
                        width < min_width
                        or height < min_height
                        or pixels < min_pixels
                    ):
                        skipped_small += 1
                        continue

                    aspect_ratio = width / height
                    if not min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                        skipped_aspect += 1
                        continue

                    image_path = output_dir / f"page_{page_index}_image_{image_index}.png"
                    pix.save(str(image_path))
                    image_paths.append(str(image_path))
                    print(
                        f"Saved {image_path} | {width}x{height} pixels | "
                        f"placements={len(rects)}"
                    )

                except Exception as exc:
                    extraction_errors += 1
                    print(
                        f"Could not extract image on page {page_index}, "
                        f"object {image_index}: {exc}"
                    )

    print("\nImage extraction summary")
    print(f"Image objects found: {total_objects}")
    print(f"Images saved: {len(image_paths)}")
    print(f"No page placement: {skipped_no_placement}")
    print(f"Too small: {skipped_small}")
    print(f"Rejected by aspect ratio: {skipped_aspect}")
    print(f"Extraction errors: {extraction_errors}")
    return image_paths


print("\n[4/9] EXTRACTING AND FILTERING PDF FIGURES...")
image_paths = extract_pdf_images(PDF_PATH)
print("Filtered image count:", len(image_paths))


# ============================================================
# 5. CREATE CLIP EMBEDDINGS, REBUILD COLLECTION, RETRIEVE FIGURES
# ============================================================
print("\n[5/9] REBUILDING THE CLIP IMAGE COLLECTION...")
image_collection = rebuild_collection(chroma_client, IMAGE_COLLECTION_NAME)
retrieved_image_paths = []
retrieved_image_distances = []
clip_model = None

if image_paths:
    clip_model = SentenceTransformer("sentence-transformers/clip-ViT-B-32")

    pil_images = []
    try:
        for image_path in image_paths:
            with Image.open(image_path) as image:
                pil_images.append(image.convert("RGB"))

        image_embeddings = clip_model.encode(
            pil_images,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    finally:
        for image in pil_images:
            image.close()

    image_metadata = []
    for path in image_paths:
        match = re.search(r"page_(\d+)_image_(\d+)", Path(path).stem)
        image_metadata.append(
            {
                "type": "figure",
                "source": path,
                "source_pdf": pdf_file.name,
                "page": int(match.group(1)) if match else 0,
            }
        )

    image_collection.upsert(
        ids=[Path(path).stem for path in image_paths],
        documents=image_paths,
        embeddings=image_embeddings.tolist(),
        metadatas=image_metadata,
    )
    print("Images stored in ChromaDB:", image_collection.count())

    query_image_embedding = clip_model.encode(
        [QUERY],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    image_results = image_collection.query(
        query_embeddings=[query_image_embedding.tolist()],
        n_results=min(TOP_K_IMAGES, len(image_paths)),
    )
    retrieved_image_paths = image_results["documents"][0]
    retrieved_image_distances = image_results["distances"][0]

    print("\nRetrieved figures (lower cosine distance is better):")
    for path, distance in zip(retrieved_image_paths, retrieved_image_distances):
        print(f"{path} | distance={distance:.4f}")
else:
    print("No suitable images were extracted; continuing without visual evidence.")


# ============================================================
# 6. EXTRACT TABLES AND RETRIEVE RELEVANT TABLES
# ============================================================

def extract_pdf_tables(pdf_path):
    """Extract real tables and add fallbacks for the two population tables."""

    extracted_tables = []

    with pymupdf.open(str(pdf_path)) as doc:
        for page_number, page in enumerate(doc, start=1):

            page_text = page.get_text("text") or ""
            upper_text = page_text.upper()

            # Fallback: these two tables are present as aligned PDF text,
            # but PyMuPDF's table detector may fail to recognize them.
            if "TABLE 12.1" in upper_text and "GROWTH" in upper_text:
                extracted_tables.append({
                    "text": (
                        f"Source: {Path(pdf_path).name}\n"
                        f"Page: {page_number}, Table: 12.1 - POPULATION\n"
                        f"{page_text}"
                    ),
                    "page": page_number,
                    "table": 1,
                })
                continue

            if (
                "TABLE 12.2" in upper_text
                and "RURAL" in upper_text
                and "URBAN" in upper_text
            ):
                extracted_tables.append({
                    "text": (
                        f"Source: {Path(pdf_path).name}\n"
                        f"Page: {page_number}, "
                        "Table: 12.2 - POPULATION IN RURAL / URBAN AREAS\n"
                        f"{page_text}"
                    ),
                    "page": page_number,
                    "table": 2,
                })
                continue

            # Normal extraction for other pages.
            try:
                found_tables = page.find_tables()
            except Exception:
                continue

            for table_number, table in enumerate(
                found_tables.tables, start=1
            ):
                rows = table.extract()

                if not rows or len(rows) < 3:
                    continue

                # Remove obvious paragraph/layout false positives.
                column_count = max(
                    (len(row) for row in rows),
                    default=0,
                )

                if column_count < 3:
                    continue

                populated_columns = sum(
                    any(
                        column < len(row)
                        and row[column] is not None
                        and str(row[column]).strip()
                        for row in rows
                    )
                    for column in range(column_count)
                )

                if populated_columns < 3:
                    continue

                lines = []
                for row in rows:
                    cells = [
                        str(cell).strip() if cell is not None else ""
                        for cell in row
                    ]
                    lines.append(" | ".join(cells))

                table_text = (
                    f"Source: {Path(pdf_path).name}\n"
                    f"Page: {page_number}, Table: {table_number}\n"
                    + "\n".join(lines)
                )

                extracted_tables.append({
                    "text": table_text,
                    "page": page_number,
                    "table": table_number,
                })

    print(f"Tables extracted: {len(extracted_tables)}")
    print(
        "Population tables found:",
        [
            item["page"]
            for item in extracted_tables
            if item["page"] in (16, 17)
        ],
    )

    return extracted_tables


# ============================================================
# 6. EXTRACT, STORE, AND RETRIEVE TABLES
# ============================================================

print("\n[6/9] EXTRACTING AND INDEXING TABLES...")

# 1. Extract table candidates from the PDF
tables = extract_pdf_tables(PDF_PATH)

# Keep the extracted table text
table_texts = [item["text"] for item in tables]

# 2. Rebuild the dedicated ChromaDB collection
# This removes old/incorrect table records.
table_collection = rebuild_collection(
    chroma_client,
    TABLE_COLLECTION_NAME,
)

# This variable will contain retrieved evidence, not every extracted table.
table_evidence = []

if table_texts:

    # 3. Create embeddings for extracted table content
    table_embeddings = embedding_model.encode(
        table_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # 4. Store extracted tables and their source metadata
    table_collection.upsert(
        ids=[
            f"table_{item['page']}_{item['table']}"
            for item in tables
        ],
        documents=table_texts,
        embeddings=table_embeddings.tolist(),
        metadatas=[
            {
                "type": "table",
                "source_pdf": Path(PDF_PATH).name,
                "page": item["page"],
                "table_number": item["table"],
            }
            for item in tables
        ],
    )

    print("Stored table records:", table_collection.count())

    # 5. Embed the user's question
    query_embedding = embedding_model.encode(
        [QUERY],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    # 6. Retrieve the most relevant table records
    results = table_collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(5, table_collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    # 7. Preserve page numbers and table identifiers
    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        table_evidence.append({
            "page": metadata.get("page"),
            "table_number": metadata.get("table_number"),
            "text": document,
            "distance": float(distance),
        })

    print("\nRetrieved tables:")

    for item in table_evidence:
        print(
            f"\nPage {item['page']} | "
            f"Table {item['table_number']} | "
            f"Distance: {item['distance']:.4f}"
        )
        print(item["text"][:700])

else:
    print("No tables were extracted.")

# ============================================================
# 7. USE THE VLM TO INTERPRET RETRIEVED FIGURES
# ============================================================
def image_to_data_url(image_path):
    """Convert a local image to a data URL accepted by the vision API."""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


vision_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY,
)


def analyze_images_with_vlm(query, paths, max_images=MAX_IMAGES_FOR_VLM):
    """Describe relevant evidence from figures without guessing unreadable values."""
    descriptions = []

    for image_path in paths[:max_images]:
        prompt = f"""You are analyzing one figure extracted from a research paper.
User question: {query}

First say whether this figure appears relevant to the question.
If it is a chart, inspect the title, axes, labels, units, legend, dates and visible values.
Report only values that can actually be read from the image. Never guess numbers from
unclear bars, lines or shapes. If values are unreadable, say that clearly. Briefly explain
what the figure shows and how it relates to the question. Separate visible evidence from
any uncertainty. Do not use outside knowledge to fill missing chart information."""

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(image_path)},
                },
            ]
        )
        response = vision_llm.invoke([message])
        descriptions.append(
            {"image": image_path, "description": response.content}
        )

    return descriptions


print("\n[7/9] ANALYZING RETRIEVED FIGURES WITH THE VLM...")
visual_descriptions = analyze_images_with_vlm(QUERY, retrieved_image_paths)
for item in visual_descriptions:
    print("\nFigure:", item["image"])
    print("VLM:", item["description"])


# ============================================================
# 8. ASSEMBLE GROUNDED MULTIMODAL EVIDENCE
# ============================================================
text_evidence = [
    f"Page {item['page']}: {item['text']}"
    for item in text_results
]
figure_evidence = [
    {
        "source": item["image"],
        "description": item["description"],
    }
    for item in visual_descriptions
]

multimodal_context = {
    "text_evidence": text_evidence,
    "table_evidence": table_evidence,
    "figure_evidence": figure_evidence,
}

print("\n[8/9] MULTIMODAL EVIDENCE SUMMARY")
print(json.dumps(multimodal_context, ensure_ascii=False, indent=2))


# ============================================================
# 9. FINAL ANSWER FROM THE MULTIMODAL CONTEXT
# ============================================================
answer_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    api_key=OPENAI_API_KEY,
)


def answer_multimodal_query(query, context):
    """Answer using retrieved evidence and be explicit when evidence is insufficient."""
    prompt_template = PromptTemplate(
        input_variables=["query", "context"],
        template="""You are a research assistant answering questions about a research paper.
Answer the question using only the evidence supplied below.

Rules:
- Do not invent facts, numbers, dates, units, or conclusions.
- If sources conflict or do not support an answer, explain the limitation.
- Prefer evidence that directly answers the question over loosely related material.
- Use page numbers from text/table evidence when available.
- A figure description may be uncertain; do not treat unreadable values as facts.
- Clearly distinguish what the paper states from what cannot be confirmed.
- Keep the answer clear and focused.

Question:
{query}

Retrieved multimodal evidence (text, tables, and figure interpretations):
{context}

Grounded answer:
""",
    )

    prompt = prompt_template.format(
        query=query,
        context=json.dumps(context, ensure_ascii=False, indent=2),
    )
    return answer_llm.invoke(prompt)


print("\n[9/9] GENERATING FINAL MULTIMODAL ANSWER...")
final_response = answer_multimodal_query(QUERY, multimodal_context)
print(f"\nQuestion: {QUERY}\n")
print(final_response.content)
