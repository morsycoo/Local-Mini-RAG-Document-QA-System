# Local Mini RAG Document Question Answering System

A local-first Retrieval-Augmented Generation (RAG) project that builds a document knowledge base through ingestion, structure-aware chunking, multilingual embeddings, and persistent vector storage. Semantic retrieval is the next implementation stage.

> **Current status:** Day 3 completed. Retrieval relevance and answer quality have not yet been evaluated.

## Project Overview

```text
Source Documents
      |
      v
Document Ingestion & Text Extraction
      |
      v
Document / Page Records (JSONL)
      |
      v
Text Chunking
      |
      v
Chunk Records (JSONL)
      |
      v
Multilingual E5 Embeddings
      |
      v
Persistent ChromaDB Collection
      |
      v
Semantic Retrieval                 <- Next stage
      |
      v
RAG Context Construction
      |
      v
LLM Answer Generation
```

The current implementation is local and does not require a hosted vector database or cloud inference service.

## Implemented Stages

### Day 1 — Document Loading & Text Extraction

Implemented a reproducible ingestion layer for PDF, Markdown, and plain-text sources.

- Recursive discovery of supported files with safety checks.
- Native text extraction from `.pdf`, `.md`, and `.txt`.
- Page-level records for PDFs and logical-page records for TXT/Markdown.
- Pydantic validation for document, page, and ingestion-result records.
- Deterministic document IDs and SHA-256 source/content hashes.
- Duplicate, unchanged, and updated-document detection.
- JSONL persistence with atomic per-file replacement and rollback handling.
- CLI entry point and automated tests.

**Recorded validation:** 96 ingestion tests passed; the full project suite recorded at the end of Day 1 reported 462 passing tests.

**Boundary:** PDF extraction is native-text extraction, not OCR. Scanned/image-only PDFs and complex layout reconstruction are not covered.

### Day 2 — Text Chunking & Document Structure Preservation

Implemented and compared two chunking strategies using the processed page records from Day 1.

- Fixed-window chunking as a baseline.
- Markdown-aware recursive chunking.
- Parsing of headings, paragraphs, lists, tables, fenced code blocks, and horizontal rules.
- Preservation of source offsets, page numbers, document IDs, section paths, and chunking configuration metadata.
- Pydantic `ChunkRecord` validation.
- Chunk persistence and reload validation.

**Recorded corpus:** 6 Markdown documents, 6 page records, and 118,204 source characters.

| Metric | Fixed-window | Recursive Markdown |
|---|---:|---:|
| Generated chunks | 119 | 228 |
| Source coverage | 100.00% | 100.00% |
| Structural preservation | 94.46% | 98.78% |
| Integrity errors | 0 | 0 |
| Median processing time | 0.000070 s | 0.011590 s |

`recursive_markdown` was selected as the primary strategy based on the recorded corpus experiment. Fixed-window remains available as a baseline.

**Recorded validation:** 53 chunking tests passed; the full project suite recorded at the end of Day 2 reported 515 passing tests.

### Day 3 — Embeddings & Local Vector Database

Implemented the embedding-generation and persistent vector-storage layer.

**Embedding model**

- Model: `intfloat/multilingual-e5-base`
- Pinned revision: `ea6fe3baa7f6bfe27fbc99fde19d01b3e2bca610`
- Embedding dimension: 768
- Data type: `float32`
- L2 normalization: enabled
- Passage prefix: `passage: `
- Query prefix for retrieval: `query: `
- Maximum sequence length: 512 tokens

**Embedding results**

| Metric | Recorded result |
|---|---:|
| Source chunks | 228 |
| Generated embeddings | 228 |
| Embedding matrix | `(228, 768)` |
| Zero vectors | 0 |
| Non-finite values | 0 |
| Mean L2 norm | 1.0 |
| Maximum complete passage input | 452 tokens |
| Token-limit overflows | 0 |

Embeddings and their manifest are persisted under:

```text
data/knowledge_base/processed/embeddings/
├── embeddings.npz
└── manifest.json
```

**Vector database**

- ChromaDB persistent client.
- Collection: `hadota_chunks_e5_base_v1`
- Distance metric: cosine.
- Embedding function: `None` (vectors are generated externally by the pinned E5 model).
- Persistent storage: `data/knowledge_base/vector_store/chroma/`.

**Recorded indexing and integrity results**

- 228 chunks indexed with embeddings and metadata.
- Collection successfully reopened through a new persistent client.
- 228 stored IDs; no missing or unexpected IDs.
- Re-indexing left the collection at 228 records, with no duplicate records created.
- Three sampled records were inspected for text, metadata, and embedding consistency.

**Recorded validation:** 146 dedicated vector-store tests passed. The recorded full project test run reported 727 passed, 0 failed, and 23 ChromaDB deprecation warnings. Code coverage was not measured.

**Important limitation:** successful indexing and integrity checks do not establish retrieval relevance or answer quality.

## Repository Artifacts

```text
data/
└── knowledge_base/
    ├── raw/                         # Original source documents
    ├── processed/
    │   ├── documents.jsonl
    │   ├── pages.jsonl
    │   ├── chunks.jsonl
    │   └── embeddings/
    │       ├── embeddings.npz
    │       └── manifest.json
    └── vector_store/
        └── chroma/                  # Persistent local ChromaDB data

app/
├── ingestion/
├── chunking/
├── embeddings/
└── vector_store/

notebooks/
├── 01_document_loading_and_text_extraction.ipynb
├── 02_text_chunking.ipynb
└── 03_embeddings_and_vector_database.ipynb

reports/
└── 3_embedding_report.json

tests/
├── ingestion/
├── chunking/
├── embeddings/
└── vector_store/
```

This is a high-level map of relevant project components. Generated local data and the ChromaDB persistence directory may be excluded from version control.

## Technology Stack

- **Language:** Python 3.13
- **Data validation:** Pydantic
- **PDF extraction:** PyMuPDF
- **Embeddings:** Sentence Transformers, `intfloat/multilingual-e5-base`
- **Vector database:** ChromaDB
- **Numerical processing:** NumPy
- **Testing:** pytest
- **Experimentation:** Jupyter Notebook

## Current Scope and Limitations

The project currently provides document ingestion, chunking, embedding generation, and persistent vector indexing.

Not yet implemented or evaluated:

- Semantic query retrieval and ranked search results.
- Retrieval metrics such as Recall@K, Precision@K, MRR, or NDCG.
- Similarity-threshold selection and retrieval-latency benchmarking.
- Hybrid lexical/vector retrieval or reranking.
- RAG context construction and LLM answer generation.
- Production deployment.

The current corpus is small (6 documents and 228 chunks). Results should not be generalized to large or diverse knowledge bases without further evaluation.

## Next Stage — Semantic Retrieval & Similarity Search

The next stage will reuse the existing artifacts and persistent ChromaDB collection:

1. Validate a non-empty user query.
2. Encode the query using the same pinned E5 model revision, with the `query: ` prefix.
3. Validate a normalized 768-dimensional query vector.
4. Query the existing ChromaDB collection using cosine distance and configurable Top-K.
5. Return ranked chunks with text, source identifiers, metadata, and distance.
6. Add automated tests and qualitative retrieval experiments.

The existing document embeddings and ChromaDB collection should not be regenerated or reset without a separately justified migration.

## Reproducibility

Run commands from the repository root using the project's configured Python environment.

```powershell
python -m pytest -q
```

Open the notebooks in order:

```text
notebooks/01_document_loading_and_text_extraction.ipynb
notebooks/02_text_chunking.ipynb
notebooks/03_embeddings_and_vector_database.ipynb
```

For exact environment versions, experiment metrics, artifact hashes, and validation details, refer to the corresponding Day 1, Day 2, and Day 3 documentation and `reports/3_embedding_report.json`.

## Project Principle

Each stage consumes validated outputs from the previous stage. Existing ingestion, chunking, embedding, and vector-store artifacts are reused rather than rebuilt unnecessarily. Experimental results are reported as measured; successful infrastructure validation is not presented as proof of retrieval or answer quality.
