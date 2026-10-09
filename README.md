# Multimodal AI Research Assistant

This project is a research-paper assistant that lets you ask questions about a PDF and get answers based on the information retrieved from it.

I built it to explore how a RAG system can work with different types of information in a research document, including text, tables, and figures.

## What it does

- Extracts text from PDF pages and splits it into smaller chunks.
- Uses embeddings and ChromaDB to store and retrieve relevant text.
- Combines semantic search with BM25 retrieval and reranking.
- Extracts and searches tables from the document.
- Retrieves relevant figures and uses a vision model to describe them when API access is available.
- Generates answers using the retrieved evidence.
- Provides a Streamlit interface connected to a FastAPI backend.
- Includes a retrieval evaluation script to measure how well relevant pages are found.

## Technologies used

- Python
- FastAPI
- Streamlit
- ChromaDB
- Sentence Transformers
- BGE-M3 embeddings
- BM25 and reciprocal rank fusion
- BGE reranker
- CLIP for figure retrieval
- PyMuPDF and pypdf
- OpenAI API for answer generation and figure descriptions

## How it works

1. The PDF is processed and its text, tables, and figures are extracted.
2. Text and table content are converted into embeddings and stored in ChromaDB. Figure embeddings are stored separately.
3. When a user asks a question, the system retrieves relevant evidence from the available collections.
4. The retrieved evidence is sent to the language model to generate an answer.
5. The answer and retrieved evidence are shown in the Streamlit interface.

## Run locally

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Install dependencies

This project uses `uv` for dependency management.

```bash
uv sync
```

If you do not have a `pyproject.toml` and lock file in the repository, install the packages listed in your dependency file instead.

### 3. Configure the API key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

Do not commit your real API key to GitHub.

### 4. Prepare the data

Place the research PDF at the path configured in the main pipeline. Run the pipeline to extract the content and create the ChromaDB collections.

```bash
uv run python complete_multimodal_ai_research_assistant.py
```

You only need to run the indexing pipeline again when you want to rebuild or update the stored document data.

### 5. Start the FastAPI backend

Open a terminal in the project folder and run:

```bash
uv run uvicorn api:app --reload
```

The API documentation should be available at:

http://127.0.0.1:8000/docs

### 6. Start the Streamlit frontend

Open another terminal and run:

```bash
uv run streamlit run streamlit_app.py
```

Use the local URL printed in the terminal to open the application.

## Evaluate retrieval

The evaluation script checks whether relevant pages appear in the retrieved results.

```bash
uv run python rag_evaluation.py
```

It reports metrics such as Hit@5, Precision@5, Recall@5, and MRR@5. Answer-quality scores need to be reviewed separately against the reference answers.

## Notes

- The PDF and generated ChromaDB data may not be included in this repository. Make sure the required files are available before running the app.
- Answer generation and figure descriptions require a working OpenAI API connection and may incur API costs.
- Table and figure extraction can vary depending on the layout and quality of the PDF.
- Retrieval metrics measure how well evidence is found; they do not by themselves prove that every generated answer is correct.

## Future improvements

- Improve table and figure understanding.
- Add more test questions and answer-quality evaluation.
- Improve error handling and deployment.
- Support multiple research papers.

---

Built as a learning project to explore retrieval-augmented generation with text, tables, and figures.
