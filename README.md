# Local Mini RAG Document QA System

> **Project status:** In progress  
> **Current milestone:** RAG Fundamentals  
> **Completed:** Document Loading & Text Extraction and Text Chunking 
> **Execution model:** Local-first; source processing runs on the developer's machine.

HADOTA-RAG is a local document question-answering project being built around the technical knowledge base of **HADOTA (حَدُّوتَة)**. The goal is to build the RAG pipeline step by step: ingest source documents, preserve their structure, create traceable chunks, generate multilingual embeddings, retrieve relevant evidence, and eventually produce answers grounded in those sources.

This repository is a work in progress. It currently contains the document ingestion and chunking layers. **It does not yet implement embeddings, vector search, retrieval, or LLM-generated answers.**

---

## Current capabilities

### 1. Document loading and text extraction — complete

The ingestion layer discovers supported files, extracts text, validates records, tracks source provenance, and persists the result for later processing.

- Recursive discovery of `.pdf`, `.md`, and `.txt` files.
- UTF-8 and UTF-8 BOM-aware text loading.
- Native-text PDF extraction with PyMuPDF, retaining PDF page boundaries.
- Pydantic validation for document, page, and ingestion-result records.
- Deterministic document IDs and SHA-256 source/content hashes.
- Detection of new, updated, unchanged, and duplicate-content documents.
- JSONL persistence for document and page records.
- Failure handling and tested rollback behavior for the two JSONL stores.
- A local CLI and automated tests.

**Observed corpus snapshot:** six Markdown source documents, 118,204 extracted characters, six logical page records, and zero extraction failures. Re-running ingestion classified all six files as unchanged.

> PDF extraction currently reads embedded/native text. Scanned or image-only PDFs are not OCR-processed. Complex PDF layouts and tables may require separate handling.

### 2. Text chunking — complete

The chunking layer consumes the processed page records and creates validated, source-traceable `ChunkRecord` objects.

Two strategies were implemented and compared:

- **Fixed-window:** character-based windows with a 180-character overlap.
- **Recursive Markdown:** structure-aware chunking using Markdown blocks and heading context.

The primary strategy selected for the current corpus is **`recursive_markdown`**. The fixed-window strategy remains available as a baseline.

Each chunk retains provenance fields such as:

- `document_id` and `page_number`
- `chunk_index`
- `char_start` and `char_end`
- `section_path`
- `content_hash`
- `strategy`, `config_signature`, and `chunking_version`

Chunk records are saved in `data/knowledge_base/processed/chunks.jsonl`, while the strategy comparison is saved in `reports/chunking_comparison.json`.

### Experiment results

Both strategies were evaluated against the same six page records and 118,204 source characters.

| Metric | Fixed-window | Recursive Markdown |
|---|---:|---:|
| Generated chunks | 119 | 228 |
| Mean chunk size | 1,164.24 chars | 518.44 chars |
| Source coverage | 100.00% | 100.00% |
| Structural blocks preserved | 699 / 740 | 731 / 740 |
| Structural preservation | 94.46% | 98.78% |
| Integrity errors | 0 | 0 |
| Median processing time | 0.000070 s | 0.011590 s |

The recursive strategy produced more, smaller chunks and preserved a larger percentage of the detected Markdown blocks. It was selected for the current knowledge base based on these measured results. Structural preservation is a document-structure metric; it does not by itself prove retrieval quality or answer accuracy.

**Important implementation notes:**

- The recursive API does not currently expose an overlap parameter; the 180-character overlap applies only to fixed-window chunking.
- Nineteen recursive chunks were shorter than 100 characters. They were retained to avoid merging across structural boundaries. Their usefulness will be evaluated during retrieval work.
- The current measured comparison corpus contains six Markdown documents. The existence of TXT/PDF extraction support does not mean the chunking comparison has been validated across a large or diverse PDF/TXT corpus.

---

## Pipeline: current state and target

### Implemented

```text
Source Documents
      |
      v
Document Discovery
      |
      v
Text Extraction
      |
      v
Validated Document / Page Records
      |
      v
JSONL Persistence
      |
      v
Fixed-window and Recursive Markdown Chunking
      |
      v
Validated Chunk Records
      |
      v
chunks.jsonl
```

### Planned next stages

```text
chunks.jsonl
      |
      v
Multilingual Embedding Generation
      |
      v
Local Vector Store
      |
      v
Similarity Search and Metadata Filtering
      |
      v
Retrieved Evidence
      |
      v
Grounded Answer Generation
      |
      v
Document Question Answering
```

The planned stages are not represented as completed functionality yet.

---

## Repository structure

The following is the current project structure relevant to the completed ingestion and chunking work. Additional application files may exist outside this view.

```text
HADOTA-RAG/
├── app/
│   ├── ingestion/
│   │   ├── models.py
│   │   ├── discovery.py
│   │   ├── extractors.py
│   │   ├── pipeline.py
│   │   └── cli.py
│   └── chunking/
│       ├── models.py
│       ├── strategies.py
│       ├── markdown.py
│       ├── markdown_parser.py
│       └── recursive.py
├── data/
│   └── knowledge_base/
│       ├── raw/                    # Original source documents
│       └── processed/
│           ├── documents.jsonl
│           ├── pages.jsonl
│           └── chunks.jsonl
├── docs/
│   ├── README_01_document_loading_and_text_extraction.md
│   └── README_02_text_chunking.md
├── notebooks/
│   ├── 01_document_loading_and_text_extraction.ipynb
│   └── 02_text_chunking.ipynb
├── reports/
│   └── chunking_comparison.json
└── tests/
    ├── ingestion/
    └── chunking/
```

---

## Reproducing the current work

Run commands from the repository root using the project's configured Python environment.

### Run document ingestion

```bash
python -m app.ingestion.cli --help
python -m app.ingestion.cli
```

The CLI's `--help` output is the authoritative reference for its available options and configured paths.

### Run tests

```bash
python -m pytest tests/ingestion/ -v
python -m pytest tests/chunking/ -v
python -m pytest -q
```

The recorded results at the end of Day 2 were:

| Test run | Result |
|---|---:|
| Ingestion tests | 96 passed |
| Chunking tests | 53 passed |
| Full project regression | 515 passed |

These are historical results from the recorded development runs. Re-run the commands after code or dependency changes to verify the current checkout.

---

## Technology

| Area | Technology |
|---|---|
| Language | Python 3.13 |
| Data validation | Pydantic |
| PDF text extraction | PyMuPDF |
| Persistence for ingestion/chunk outputs | JSON Lines (JSONL) |
| Tests | pytest |
| Notebook experiments | Jupyter Notebook |

No embedding model or vector database is part of the implemented pipeline at this stage.

---

## Current limitations

- No OCR for scanned/image-only PDFs.
- No dedicated layout reconstruction for complex PDF columns or tables.
- Recursive Markdown chunking currently has no configurable overlap.
- Small structure-preserving chunks have not yet been evaluated through retrieval.
- JSONL persistence is suitable for this local learning project but does not provide a transactional multi-writer database.
- No embedding generation, vector index, similarity search, retrieval, reranking, or generated answers yet.
- No production deployment or cloud service is claimed.

---

## Roadmap

- [x] Document Loading & Text Extraction
- [x] Text Chunking & Document Structure Preservation
- [ ] Embeddings & Vector Database
- [ ] Retrieval Pipeline
- [ ] RAG Generation & Source Attribution
- [ ] Evaluation & HADOTA Integration

---

## Project objective

The objective is not merely to connect an LLM to a vector database. The project is designed to make each stage observable and testable: source extraction, provenance, chunk quality, embedding generation, retrieval relevance, and answer grounding. Each later stage should be validated against explicit experiments rather than treated as successful simply because the pipeline runs.

**Current milestone:** a tested local document-ingestion and Markdown-aware chunking foundation, ready for embedding experiments.
